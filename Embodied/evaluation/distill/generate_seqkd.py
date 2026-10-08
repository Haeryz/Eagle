"""Sequence-level self-distillation data for the MTP stream.

The frozen base model decodes COCO train2017 images in Slow (pure NTP) mode with greedy decoding; its outputs become
the training targets (sequence-level KD, Kim & Rush, EMNLP 2016). Distilled targets follow one deterministic object
order and spacing, so they are less multimodal than raw annotations (Zhou, Gu & Neubig, ICLR 2020), which is the
property the parallel MTP block needs. Following dParallel (ICLR 2026), responses with wrong answers are filtered:
a sample is kept only if its predicted boxes match COCO ground truth (per-category Hungarian matching, IoU >= 0.5)
with precision >= --min_precision.

Images used by any evaluation set (see evaluation/tools/make_subsets.py) are excluded.
Output is resumable: already processed image ids are skipped.
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

import requests
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference_grounding_ddp import LocateAnythingWorker, parse_prediction  # noqa: E402


def precision_against_gt(pred, gt, iou_thr):
    """pred/gt: {category: [[x1,y1,x2,y2], ...]} in pixels. Returns (num_correct, num_pred)."""
    correct, total = 0, 0
    for cat, boxes in pred.items():
        boxes = [b for b in boxes if len(b) == 4]
        total += len(boxes)
        if not boxes or not gt.get(cat):
            continue
        iou = box_iou(torch.tensor(boxes, dtype=torch.float), torch.tensor(gt[cat], dtype=torch.float))
        rows, cols = linear_sum_assignment(-iou.numpy())
        correct += int((iou[rows, cols] >= iou_thr).sum())
    return correct, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--coco_ann", required=True, help="instances_train2017.json")
    ap.add_argument("--excluded_ids", required=True, help="excluded_coco_ids.json from make_subsets.py")
    ap.add_argument("--image_dir", required=True, help="where train2017 images are cached")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--num_images", type=int, default=3000)
    ap.add_argument("--min_precision", type=float, default=0.8)
    ap.add_argument("--iou_thr", type=float, default=0.5)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    coco = json.load(open(args.coco_ann))
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    excluded = set(json.load(open(args.excluded_ids)))
    gt = defaultdict(lambda: defaultdict(list))
    for a in coco["annotations"]:
        if a["iscrowd"] or a["image_id"] in excluded:
            continue
        x, y, w, h = a["bbox"]
        gt[a["image_id"]][cat_name[a["category_id"]]].append([x, y, x + w, y + h])
    image_ids = sorted(gt)
    random.Random(args.seed).shuffle(image_ids)
    image_ids = image_ids[:args.num_images]
    del coco

    done = set()
    if os.path.exists(args.out_jsonl):
        done = {json.loads(l)["image_id"] for l in open(args.out_jsonl)}
    os.makedirs(args.image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    worker = LocateAnythingWorker(args.model_path, generation_mode="slow", dtype=args.dtype, seed=args.seed)
    worker.temperature = 0  # greedy teacher: the mode of the NTP distribution
    kept = 0
    with open(args.out_jsonl, "a") as fout:
        for image_id in tqdm(image_ids):
            if image_id in done:
                continue
            rel = f"coco/train2017/{image_id:012d}.jpg"
            path = os.path.join(args.image_dir, rel)
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                r = requests.get(f"http://images.cocodataset.org/train2017/{image_id:012d}.jpg", timeout=60)
                r.raise_for_status()
                open(path, "wb").write(r.content)

            image = Image.open(path).convert("RGB")
            w, h = image.size
            categories = list(gt[image_id])
            output, question = worker.generate(image, categories, max_new_tokens=args.max_new_tokens)
            response = output.replace("<|im_end|>", "").strip()
            pred = parse_prediction(response, w, h)
            correct, total = precision_against_gt(pred, gt[image_id], args.iou_thr)
            keep = total > 0 and correct / total >= args.min_precision
            kept += keep
            fout.write(json.dumps({
                "image_id": image_id,
                "image": rel,
                "keep": bool(keep),
                "precision": correct / total if total else 0.0,
                "num_pred": total,
                "num_gt": sum(len(v) for v in gt[image_id].values()),
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": response},
                ],
            }) + "\n")
            fout.flush()
    print(f"kept {kept} new samples")


if __name__ == "__main__":
    main()
