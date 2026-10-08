"""GRPO for LocateAnything's hybrid Parallel Box Decoding, rewarding accuracy *and* fewer NTP fallbacks.

Ported from the Perception-R1 reference implementation (Yu et al., NeurIPS 2025; github.com/linkangheng/PR1,
`src/open_r1/rewards.py::pr1_detection_reward` and `trainer/grpo_trainer_vllm.py::compute_loss`):
  * group of G sampled completions per prompt, group-normalized advantages,
  * per-token loss  -exp(logp - logp.detach()) * A + beta * k3-KL(policy || reference),
    reference = the same network with the LoRA adapter disabled, beta = 0.04 (TRL default used by PR1),
  * detection reward = 0.25*format + 0.75*F1 + class + mean IoU over Hungarian-matched pairs
    (PR1 matcher: L1 + GIoU + class cost, IoU > 0.5; DO_PENALTY=False as in PR1's detection script).

Adaptations to PBD (the paper's stated future work: RL to "reduce fallback frequency"):
  * Rollouts use the model's own hybrid decoding under its released sampling distribution (temperature 0.7,
    top-p 0.9; PR1's T=1.0 causes repetition loops here). Coordinates in MTP box blocks are *sampled* from the
    coordinate-restricted distribution so the group explores; the fallback rule is unchanged.
  * A second reward, r_par = 1 - (NTP fallbacks / MTP box blocks), is group-normalized separately and added to the
    accuracy advantage (the two-reward normalization of LightningRL, arXiv 2603.13319), so neither reward's scale
    has to be hand-weighted.
  * The rollout log-likelihood is computed in one teacher-forced pass with the PBD training layout: the NTP stream
    scores AR tokens, and every MTP step is re-created as an [anchor, <mask> x5] block at the anchor's positions,
    which sees exactly the prefix it saw at decode time (as in d1/diffu-GRPO, NeurIPS 2025, the likelihood of a
    parallel block is the product of its factorized per-position probabilities). Only sampled tokens are actions;
    deterministic structural tokens are excluded.
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou, generalized_box_iou

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evaluation"))
from eaglevl.train.wandb_env import load_env  # noqa: E402
from inference_compat import (apply_chat_template, build_generate_kwargs, parse_statistic_info,  # noqa: E402
                              prepare_generation_inputs, process_vision_info)
from inference_grounding_ddp import parse_bbox_with_labels  # noqa: E402
from eaglevl.utils.locany.generate_utils import top_p_logits  # noqa: E402  (same transform the sampler applies)


# ----------------------------------------------------------------------------- reward (Perception-R1 port)
def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def pr1_detection_reward(pred_boxes, pred_ids, gt_boxes, gt_ids, format_ok, iou_threshold=0.5):
    """Boxes normalized to [0, 1] (xyxy); ids are category indices. Mirrors PR1's pr1_detection_reward with
    DO_PENALTY=False (their detection setting). Returns (reward, f1, mean_iou)."""
    reward = 0.25 * float(format_ok)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return reward, 0.0, 0.0
    pb, gb = torch.tensor(pred_boxes, dtype=torch.float), torch.tensor(gt_boxes, dtype=torch.float)
    pi, gi = torch.tensor(pred_ids), torch.tensor(gt_ids)
    cost_class = -(pi[:, None] == gi[None, :]).float()
    cost_bbox = torch.cdist(box_xyxy_to_cxcywh(pb), box_xyxy_to_cxcywh(gb), p=1)
    cost_giou = -generalized_box_iou(pb, gb)
    C = 5 * cost_bbox + 1 * cost_class + 2 * cost_giou
    src, tgt = linear_sum_assignment(C.numpy())
    ious = box_iou(pb[src], gb[tgt]).diag()
    keep = ious > iou_threshold
    src, tgt, ious = src[keep.numpy()], tgt[keep.numpy()], ious[keep]
    if len(src) == 0:
        return reward, 0.0, 0.0
    precision, recall = len(src) / len(pb), len(src) / len(gb)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    cls_reward = (pi[src] == gi[tgt]).float().mean().item()
    reward += 0.75 * f1 + cls_reward + ious.mean().item()
    return reward, f1, ious.mean().item()


def parse_normalized(response, categories):
    """-> (boxes in [0,1], category ids, format_ok). format_ok: finished with <|im_end|> and parsed cleanly."""
    boxes, ids = [], []
    cat_to_id = {c: i for i, c in enumerate(categories)}
    try:
        for cat, coords, is_point in parse_bbox_with_labels(response):
            if not is_point and len(coords) == 4:
                boxes.append([c / 1000.0 for c in coords])
                ids.append(cat_to_id.get(cat, -1))
        format_ok = response.rstrip().endswith("<|im_end|>")
    except Exception:
        format_ok = False
    return boxes, ids, format_ok


# ----------------------------------------------------------------------------- rollout scoring
def build_scoring_inputs(gen, mask_id):
    """Teacher-forced PBD layout for one rollout. Returns ids, position ids and the action list
    [(hidden_index, token, kind, logp_at_sampling)]; hidden state t predicts the token at t+1 (NTP stream) or the
    token in slot j of an MTP block (hidden at block_start + j)."""
    seq = gen["prompt_ids"] + gen["generated_ids"]
    ids, pos, actions = list(seq), list(range(len(seq))), []
    t = gen["prompt_len"]
    for anchor, kinds, logps in gen["trace"]:
        if anchor is None:  # AR step: one token at full index t, predicted by the NTP hidden state at t - 1
            actions.append((t - 1, seq[t], "ar", logps[0]))
        else:
            assert anchor + 1 == t, (anchor, t)
            start = len(ids)
            ids += [seq[anchor]] + [mask_id] * 5
            pos += list(range(anchor, anchor + 6))
            for j, (k, lp) in enumerate(zip(kinds, logps)):
                if k != "det":
                    actions.append((start + j, seq[anchor + 1 + j], k, lp))
        t += len(kinds)
    return ids, pos, actions


def action_logps(lm_base, vis, image_token_index, ids, pos, actions, coord_range, device, temperature=1.0,
                 top_p=None):
    """Log-probs of the sampled actions under the exact sampling distribution: logits / temperature, then nucleus
    truncation (generate_utils.top_p_logits), then softmax; coordinates renormalized over the coordinate vocab."""
    hidden = lm_base.model(
        input_ids=torch.tensor([ids], device=device), visual_features=vis, image_token_index=image_token_index,
        position_ids=torch.tensor([pos], device=device), use_cache=False,
    ).last_hidden_state[0]
    idx = torch.tensor([a[0] for a in actions], device=device)
    tok = torch.tensor([a[1] for a in actions], device=device)
    logits = (hidden[idx] @ lm_base.lm_head.weight.t()).float() / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    lp_full = torch.log_softmax(logits, dim=-1).gather(1, tok[:, None]).squeeze(1)
    c0, c1 = coord_range
    is_coord = torch.tensor([a[2] == "mtp_coord" for a in actions], device=device)
    lp_coord = torch.log_softmax(logits[:, c0:c1 + 1], dim=-1).gather(
        1, (tok - c0).clamp(0, c1 - c0)[:, None]).squeeze(1)
    return torch.where(is_coord, lp_coord, lp_full)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="work_dirs/la3b_eval")
    ap.add_argument("--init_lora", required=True, help="adapter to start from (e.g. the E2b SFT adapter)")
    ap.add_argument("--prompts_jsonl", default="work_dirs/seqkd/raw.jsonl")
    ap.add_argument("--image_root", default="work_dirs/seqkd/images")
    ap.add_argument("--coco_ann", default="work_dirs/coco_ann/instances_train2017.json")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_generations", type=int, default=8)       # PR1
    ap.add_argument("--prompts_per_step", type=int, default=2)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.04)             # PR1 / TRL default, k3 KL
    # Rollouts use the model's released sampling distribution (temperature 0.7, top-p 0.9; the eval decoding).
    # PR1's T=1.0 suits Qwen2.5-VL but drives this model into repetition loops (diagnosed: F1 0.03, 0% clean
    # finishes, 954 tokens/rollout at T=1.0 vs F1 0.80, 100%, 54 tokens at T=0.7/top-p 0.9).
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_gt_boxes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--check_consistency", action="store_true",
                    help="only verify that teacher-forced scoring reproduces sampling-time log-probs, then exit")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    from peft import PeftModel
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=dtype).to(device)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    model.language_model = PeftModel.from_pretrained(model.language_model, args.init_lora, is_trainable=True)
    model.eval()
    lm = model.language_model                       # PeftModelForCausalLM
    lm_base = lm.base_model.model                   # Qwen2ForCausalLM with LoRA layers injected
    lm_base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.float()
    tok_ids = model.token_ids
    coord_range = (tok_ids["coord_start_token_id"], tok_ids["coord_end_token_id"])
    mask_id = tok_ids["default_mask_token_id"]

    # prompts: the self-distillation image pool (disjoint from every eval set) with COCO GT
    coco = json.load(open(args.coco_ann))
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    gt = defaultdict(lambda: defaultdict(list))
    sizes = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}
    for a in coco["annotations"]:
        if not a["iscrowd"]:
            x, y, w, h = a["bbox"]
            gt[a["image_id"]][cat_name[a["category_id"]]].append([x, y, x + w, y + h])
    del coco
    pool = []
    for row in map(json.loads, open(args.prompts_jsonl)):
        g = gt[row["image_id"]]
        if 0 < sum(len(v) for v in g.values()) <= args.max_gt_boxes:
            pool.append((row, g))
    random.Random(args.seed).shuffle(pool)

    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: 1 - s / args.steps)   # TRL default: linear decay
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    run = None
    if not args.check_consistency:
        load_env()
        import wandb
        run = wandb.init(project=os.environ["WANDB_PROJECT"], name=os.path.basename(args.output_dir.rstrip("/")),
                         job_type="grpo", config=vars(args))
    os.makedirs(args.output_dir, exist_ok=True)

    def rollout(image, question):
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        image_inputs, video_inputs = process_vision_info(processor, messages)
        inputs = processor(text=[apply_chat_template(processor, messages)], images=image_inputs,
                           videos=video_inputs, return_tensors="pt", padding=True)
        prepared = prepare_generation_inputs(inputs, device)
        kw = build_generate_kwargs(prepared, processor, "hybrid", args.max_new_tokens, include_eos_token=True,
                                   verbose=True, temperature=args.temperature)
        # no repetition penalty: it would make token log-probs depend on the whole history
        kw.update(top_p=args.top_p, repetition_penalty=1.0, return_trace=True, sample_coords=True)
        return prepared, kw

    cursor = 0
    for step in range(1 if args.check_consistency else args.steps):
        t0 = time.time()
        logs = defaultdict(list)
        optim.zero_grad(set_to_none=True)
        n_rollouts = args.prompts_per_step * args.num_generations
        for _ in range(args.prompts_per_step):
            row, g = pool[cursor % len(pool)]
            cursor += 1
            image = Image.open(os.path.join(args.image_root, row["image"])).convert("RGB")
            w, h = image.size
            question = row["conversations"][0]["value"]
            categories = list(g)
            gt_boxes = [[b[0] / w, b[1] / h, b[2] / w, b[3] / h] for c in categories for b in g[c]]
            gt_ids = [i for i, c in enumerate(categories) for _ in g[c]]

            prepared, kw = rollout(image, question)
            grid = prepared["image_grid_hws"]
            if not torch.is_tensor(grid):  # the processor returns numpy; generate() converts it the same way
                grid = torch.from_numpy(grid).to(device, dtype=torch.int32)
            with torch.no_grad():
                vit = model.extract_feature(prepared["pixel_values"].to(dtype), grid)
                vis = model.mlp1(torch.cat(vit, dim=0))
            gens, r_acc, r_par = [], [], []
            for _ in range(1 if args.check_consistency else args.num_generations):
                with torch.no_grad():
                    out = model.generate(**kw)
                response, stats, info = out[0], parse_statistic_info(out[:3]), out[3]
                info["prompt_ids"] = prepared["input_ids"][0].tolist()
                boxes, ids, format_ok = parse_normalized(response, categories)
                rew, f1, miou = pr1_detection_reward(boxes, ids, gt_boxes, gt_ids, format_ok)
                fallback = stats["switch_to_ar"] / max(stats["num_box_blocks"], 1)
                gens.append(info)
                r_acc.append(rew)
                r_par.append(1.0 - fallback)
                logs["reward_acc"].append(rew); logs["f1"].append(f1); logs["miou"].append(miou)
                logs["fallback_rate"].append(fallback); logs["format_ok"].append(float(format_ok))
                logs["num_boxes"].append(stats["num_boxes"])

            if args.check_consistency:
                ids_, pos_, actions = build_scoring_inputs(gens[0], mask_id)
                lm_base.model.training = True
                with torch.no_grad():
                    lp = action_logps(lm_base, vis, model.config.image_token_index, ids_, pos_, actions,
                                      coord_range, device, args.temperature, args.top_p)  # exact sampling transform
                ref = torch.tensor([a[3] for a in actions], device=device)
                diff = (lp - ref).abs()
                by_kind = defaultdict(list)
                for a, d in zip(actions, diff.tolist()):
                    by_kind[a[2]].append(d)
                print(f"[consistency] {len(actions)} actions; max |scored - sampled| logp = {diff.max().item():.4f}, "
                      f"mean = {diff.mean().item():.4f}; by kind: "
                      + ", ".join(f"{k}: n={len(v)} max={max(v):.4f}" for k, v in by_kind.items()))
                print("[consistency] response:", response[:200])
                return

            def norm(x):
                x = torch.tensor(x)
                return (x - x.mean()) / (x.std() + 1e-4)
            adv = norm(r_acc) + norm(r_par)

            # Only the decoder's own flag: it selects the PBD block mask (and grad checkpointing); LoRA dropout and
            # other submodules stay in eval mode so scored log-probs match the sampling distribution.
            lm_base.model.training = True
            for gen, a in zip(gens, adv.tolist()):
                ids_, pos_, actions = build_scoring_inputs(gen, mask_id)
                if not actions or a == 0.0:
                    continue
                with torch.no_grad(), lm.disable_adapter():
                    # Policy and reference log-probs from raw logits, as Perception-R1's get_per_token_logps (no
                    # temperature / nucleus truncation): truncation gives tokens outside the reference nucleus
                    # log-prob ~ -1e38 and an exploding KL (observed: k3 ~ 1.5e36 at step 0).
                    ref_lp = action_logps(lm_base, vis, model.config.image_token_index, ids_, pos_, actions,
                                          coord_range, device)
                lp = action_logps(lm_base, vis, model.config.image_token_index, ids_, pos_, actions,
                                  coord_range, device)
                k3 = torch.exp(ref_lp - lp) - (ref_lp - lp) - 1
                per_token = -torch.exp(lp - lp.detach()) * a + args.beta * k3
                loss = per_token.mean() / n_rollouts
                scaler.scale(loss).backward()
                logs["kl"].append(k3.mean().item())
                logs["sampled_logp_gap"].append((lp.detach() - torch.tensor([x[3] for x in actions],
                                                                            device=device)).abs().mean().item())
            lm_base.model.training = False

        scaler.unscale_(optim)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item()
        scaler.step(optim)
        scaler.update()
        sched.step()
        summary = {k: sum(v) / len(v) for k, v in logs.items() if v}
        summary.update(step=step, lr=sched.get_last_lr()[0], grad_norm=grad_norm, step_time=time.time() - t0)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()}), flush=True)
        if run is not None:
            run.log(summary, step=step)

    lm.save_pretrained(os.path.join(args.output_dir, "llm_lora"), save_embedding_layers=False)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
