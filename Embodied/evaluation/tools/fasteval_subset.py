"""Score saved predictions with the paper's official COCO/LVIS pipeline (FastEvaluate), restricted to an eval subset.

Runs the repo's own utils/convert_coco_lvis_to_standard_format.py (--positive_only, as eval_coco.sh / eval_lvis.sh)
and fastevaluate.evaluate, exactly as metrics/coco_lvis_metric.py does. The only addition is restricting the GT json
to the images of the subset, so images that were not evaluated are not counted as misses.

Usage: python fasteval_subset.py --gt_json instances_val2017.json --eval_type coco \
           --run E0=work_dirs/results/E0/hybrid/COCO/answer.jsonl --run E3=...
"""
import argparse
import json
import os
import subprocess
import sys

import fastevaluate as fe

EVAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(EVAL_DIR, "metrics"))
from coco_lvis_metric import f1, safe_mean  # noqa: E402


def subset_gt(gt_json, pred_jsonl, out_json):
    names = {os.path.basename(json.loads(l)["image_path"]) for l in open(pred_jsonl)}
    gt = json.load(open(gt_json))
    images = [im for im in gt["images"] if os.path.basename(im["file_name"]) in names]
    ids = {im["id"] for im in images}
    gt["images"] = images
    gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] in ids]
    json.dump(gt, open(out_json, "w"))
    return len(images)


def score(pred_jsonl, gt_json, eval_type, workdir):
    os.makedirs(workdir, exist_ok=True)
    tsv = os.path.join(workdir, "fast_eval.tsv")
    subprocess.run([sys.executable, os.path.join(EVAL_DIR, "utils", "convert_coco_lvis_to_standard_format.py"),
                    "--our_pred_jsonl", pred_jsonl, "--coco_json", gt_json, "--out_tsv", tsv, "--positive_only"],
                   check=True, capture_output=True)
    res = fe.evaluate(gt_json, tsv, 0, 0, eval_type)
    out = {}
    for tag, p, r in [("mean", "precision", "recall"), ("50", "precision50", "recall50"),
                      ("95", "precision95", "recall95")]:
        out[f"f1_{tag}"] = 100 * f1(safe_mean(res.get(p, [])), safe_mean(res.get(r, [])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_json", required=True)
    ap.add_argument("--eval_type", choices=["coco", "lvis"], required=True)
    ap.add_argument("--run", action="append", required=True, help="name=answer.jsonl")
    ap.add_argument("--workdir", default="work_dirs/fasteval")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    gt_sub = None
    for spec in args.run:
        name, pred = spec.split("=", 1)
        if gt_sub is None:  # all runs share the same subset of images
            gt_sub = os.path.join(args.workdir, f"gt_subset_{args.eval_type}.json")
            os.makedirs(args.workdir, exist_ok=True)
            n = subset_gt(args.gt_json, pred, gt_sub)
            print(f"GT restricted to {n} subset images")
        rows.append({"run": name, **score(pred, gt_sub, args.eval_type, os.path.join(args.workdir, args.eval_type, name))})
    print(f"| run | F1@mean | F1@0.5 | F1@0.95 |  ({args.eval_type.upper()}, FastEvaluate)")
    for r in rows:
        print(f"| {r['run']} | {r['f1_mean']:.2f} | {r['f1_50']:.2f} | {r['f1_95']:.2f} |")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
