"""TIDE-style error breakdown of saved predictions (Bolya et al., "TIDE: A General Toolbox for Identifying Object
Detection Errors", ECCV 2020), adapted to score-free set predictions.

Every predicted box gets exactly one label, in this priority:
  correct     IoU >= 0.5 with an unmatched GT of the same category
  duplicate   IoU >= 0.5 with a same-category GT that is already matched
  class       IoU >= 0.5 with a GT of another category
  localize    0.1 <= IoU < 0.5 with a same-category GT (right object, imprecise box)
  background  everything else
Unmatched GT boxes are 'missed'. Predictions are matched in output order (the model emits no scores).
Also counts runaway samples (hit the token cap in a repetition loop).

Usage: python error_analysis.py --run E0=answer.jsonl --run E3=answer.jsonl [--out errors.json]
"""
import argparse
import json
from collections import Counter

import torch
from torchvision.ops import box_iou

RUNAWAY_TOKENS = 8000
OCR_DATASETS = {"SROIE", "TotalText", "IC15", "HierText"}


def _boxes(d):
    return {c: [b for b in v if len(b) == 4] for c, v in d.items()}


def analyze_sample(pred, gt, return_labels=False):
    """Counter of error types; with return_labels also [(category, box, label)] per prediction and the missed GT."""
    pred, gt = _boxes(pred), _boxes(gt)
    gt_list = [(c, b) for c, v in gt.items() for b in v]
    counts, labels = Counter(), []
    if not gt_list:
        labels = [(c, b, "background") for c, v in pred.items() for b in v]
        counts["background"] += len(labels)
        return (counts, labels, []) if return_labels else counts
    g = torch.tensor([b for _, b in gt_list], dtype=torch.float)
    gcat = [c for c, _ in gt_list]
    matched = [False] * len(gt_list)
    for c, boxes in pred.items():
        for b in boxes:
            iou = box_iou(torch.tensor([b], dtype=torch.float), g)[0]
            same = torch.tensor([gc == c for gc in gcat])
            s_iou = torch.where(same, iou, torch.zeros_like(iou))
            free = s_iou.clone()
            free[torch.tensor(matched)] = 0
            if free.max() >= 0.5:
                matched[int(free.argmax())] = True
                counts["correct"] += 1
                labels.append((c, b, "correct"))
            elif s_iou.max() >= 0.5:
                counts["duplicate"] += 1
                labels.append((c, b, "duplicate"))
            elif torch.where(~same, iou, torch.zeros_like(iou)).max() >= 0.5:
                counts["class"] += 1
                labels.append((c, b, "class"))
            elif s_iou.max() >= 0.1:
                counts["localize"] += 1
                labels.append((c, b, "localize"))
            else:
                counts["background"] += 1
                labels.append((c, b, "background"))
    counts["missed"] += matched.count(False)
    counts["gt"] += len(gt_list)
    if return_labels:
        return counts, labels, [gt_list[i] for i, m in enumerate(matched) if not m]
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=answer.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude_runaway", action="store_true",
                    help="drop samples that hit the token cap (each can add hundreds of repeated boxes)")
    args = ap.parse_args()
    rows = []
    for spec in args.run:
        name, path = spec.split("=", 1)
        total, runaway, n = Counter(), 0, 0
        for d in map(json.loads, open(path)):
            gt = d["gt"] if isinstance(d["gt"], dict) else {}
            pred = d.get("extracted_predictions", {})
            if d.get("dataset_name") in OCR_DATASETS:  # text detection is class-agnostic (keys are transcriptions)
                gt = {"text": [b for v in gt.values() for b in v]}
                pred = {"text": [b for v in pred.values() for b in v]}
            is_runaway = d.get("decode_stats", {}).get("num_tokens", 0) >= RUNAWAY_TOKENS
            runaway += is_runaway
            if is_runaway and args.exclude_runaway:
                continue
            total += analyze_sample(pred, gt)
            n += 1
        preds = sum(total[k] for k in ["correct", "duplicate", "class", "localize", "background"])
        row = {"run": name, "samples": n, "predictions": preds, "gt": total["gt"], "runaway": runaway}
        for k in ["correct", "duplicate", "class", "localize", "background"]:
            row[f"{k}_pct_of_pred"] = 100 * total[k] / max(preds, 1)
        row["missed_pct_of_gt"] = 100 * total["missed"] / max(total["gt"], 1)
        rows.append(row)
    keys = ["correct", "localize", "duplicate", "class", "background"]
    print("| run | preds | " + " | ".join(f"{k} %" for k in keys) + " | missed % of GT | runaway |")
    print("|" + "---|" * (len(keys) + 4))
    for r in rows:
        print(f"| {r['run']} | {r['predictions']} | " + " | ".join(f"{r[k + '_pct_of_pred']:.1f}" for k in keys)
              + f" | {r['missed_pct_of_gt']:.1f} | {r['runaway']} |")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
