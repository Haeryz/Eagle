"""Evaluate fallback criteria as *wrong-box detectors* at an equal fallback budget.

A fallback criterion should fire on MTP blocks whose box would be wrong. Using a Fast-mode run (every MTP box block
accepted, per-block features logged in decode_stats['blocks']), each block is paired with its output box (blocks and
boxes appear in the same order) and labelled correct / wrong against GT (IoU >= 0.5 with an unmatched GT of the same
category). Criteria compared:
  release   the shipped rule: any coordinate with top-1 < 0.9, >1 coordinate in the top-k and top-k spread > 60
  entropy   EB-Sampler (Ben-Hamu et al., NeurIPS 2025): the sum of the block's coordinate entropies bounds the error
            of decoding them in parallel; fire on the highest-entropy blocks
  top1      1 - min coordinate top-1 probability (confidence-threshold rule, Fast-dLLM, ICLR 2026)
For each continuous score, the threshold is set so it fires on exactly as many blocks as the release rule (equal
budget). We report the share of fired blocks that are wrong (precision), the share of wrong boxes caught (recall),
and the AUROC of the score.

Usage: python criterion_analysis.py --run E2b_fast=work_dirs/results/E2b/fast/COCO/answer.jsonl ...
"""
import argparse
import json
import os
import sys

import torch
from PIL import Image
from torchvision.ops import box_iou

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference_grounding_ddp import parse_bbox_with_labels  # noqa: E402


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    s = torch.tensor(pos)[:, None] - torch.tensor(neg)[None, :]
    return ((s > 0).float().mean() + 0.5 * (s == 0).float().mean()).item()


def collect(path, image_root):
    feats, wrong = [], []
    for d in map(json.loads, open(path)):
        blocks = d.get("decode_stats", {}).get("blocks")
        if not blocks:
            continue
        parsed = [(c, b) for c, b, is_point in parse_bbox_with_labels(d["raw_response"]) if not is_point and len(b) == 4]
        coord_blocks = [b for b in blocks if b["type"] == "coord_box"]
        if len(parsed) != len(coord_blocks):  # alignment check: skip samples with boxes from other decode paths
            continue
        w, h = Image.open(os.path.join(image_root, d["image_path"])).size
        gt = d["gt"] if isinstance(d["gt"], dict) else {}
        gt_list = [(c, [x / w * 1000, y / h * 1000, x2 / w * 1000, y2 / h * 1000]) for c, v in gt.items()
                   for x, y, x2, y2 in [bb for bb in v if len(bb) == 4]]
        matched = [False] * len(gt_list)
        for (c, b), blk in zip(parsed, coord_blocks):
            ok = False
            same = [i for i, (gc, _) in enumerate(gt_list) if gc == c and not matched[i]]
            if same:
                iou = box_iou(torch.tensor([b], dtype=torch.float),
                              torch.tensor([gt_list[i][1] for i in same], dtype=torch.float))[0]
                j = int(iou.argmax())
                if iou[j] >= 0.5:
                    matched[same[j]] = ok = True
            feats.append(blk)
            wrong.append(not ok)
    return feats, wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=answer.jsonl (Fast mode, features logged)")
    ap.add_argument("--image_root", default="work_dirs/evaldata/images")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for spec in args.run:
        name, path = spec.split("=", 1)
        feats, wrong = collect(path, args.image_root)
        if not feats:
            continue
        scores = {
            "release": [float(any(f["abnormal"])) for f in feats],
            "entropy": [sum(f["entropy"]) for f in feats],
            "top1": [1 - min(f["top1"]) for f in feats],
        }
        budget = int(sum(scores["release"]))
        n_wrong = sum(wrong)
        row = {"run": name, "blocks": len(feats), "wrong_rate": 100 * n_wrong / len(feats),
               "budget_fires": budget, "budget_pct": 100 * budget / len(feats)}
        for k, s in scores.items():
            if k == "release":
                fired = [bool(v) for v in s]
            else:  # fire on the `budget` highest-scoring blocks
                order = sorted(range(len(s)), key=lambda i: -s[i])[:budget]
                fired = [False] * len(s)
                for i in order:
                    fired[i] = True
            hits = sum(f and w for f, w in zip(fired, wrong))
            row[f"{k}_precision"] = 100 * hits / max(sum(fired), 1)
            row[f"{k}_recall"] = 100 * hits / max(n_wrong, 1)
            row[f"{k}_auroc"] = auroc(s, wrong)
        rows.append(row)
        print(f"{name}: {len(feats)} blocks, {row['wrong_rate']:.1f}% wrong; equal budget = {budget} fires "
              f"({row['budget_pct']:.1f}%)")
        for k in scores:
            print(f"   {k:8s} precision {row[k + '_precision']:5.1f}%  recall of wrong boxes {row[k + '_recall']:5.1f}%"
                  f"  AUROC {row[k + '_auroc']:.3f}")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
