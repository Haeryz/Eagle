"""Summarize fallback-reduction experiments: F1@mIoU (same computation as metrics/other_metric.py) plus the
hybrid-mode fallback statistics logged per sample in `decode_stats` by inference_grounding_ddp.py.

Usage:
    python summarize_fallback.py --run E0_hybrid=path/answer.jsonl --run E1_hybrid=path/answer.jsonl ... \
        --out results.json
"""
import argparse
import contextlib
import io
import json
import os
import sys
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metrics"))
from other_metric import UniversalMetricsCalculator  # noqa: E402

RUNAWAY_TOKENS = 8000  # outputs this long hit the max_new_tokens=8192 cap (repetition loops)
IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def f1_by_dataset(data):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        all_results = UniversalMetricsCalculator().calculate_all_metrics(data, IOU_THRESHOLDS)
    out = {}
    for iou, results in all_results.items():
        for key, m in results["basic_metrics"].items():
            if not m["recalls"]:
                continue
            p, r = mean(m["precisions"]), mean(m["recalls"])
            out.setdefault(key, {})[float(iou)] = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {k: {"f1_miou": 100 * mean(v.values()), "f1_50": 100 * v.get(0.5, 0.0)} for k, v in out.items()}


def decode_summary(data):
    stats = [d["decode_stats"] for d in data if "decode_stats" in d]
    if not stats:
        return {}
    tot = lambda k: sum(s.get(k, 0.0) for s in stats)
    blocks, coords = tot("num_box_blocks"), tot("num_coords")
    # Per-sample (macro) averages are the primary numbers: pooled ratios are dominated by the few samples that
    # run away (repeat boxes until max_new_tokens), and which samples run away is sensitive to tiny numeric noise.
    with_blocks = [s for s in stats if s.get("num_box_blocks", 0) > 0]
    with_coords = [s for s in stats if s.get("num_coords", 0) > 0]
    macro = lambda f, pool: 100 * mean(f(s) for s in pool) if pool else 0.0
    return {
        "samples": len(stats),
        "fallback_rate_macro": macro(lambda s: s["switch_to_ar"] / s["num_box_blocks"], with_blocks),
        "fallback_ambig_rate_macro": macro(lambda s: s["switch_ambig"] / s["num_box_blocks"], with_blocks),
        "fallback_format_rate_macro": macro(lambda s: s["switch_format"] / s["num_box_blocks"], with_blocks),
        "coord_entropy_macro": mean(s["sum_coord_entropy"] / s["num_coords"] for s in with_coords) if with_coords else 0.0,
        "coord_top1_macro": mean(s["sum_coord_top1"] / s["num_coords"] for s in with_coords) if with_coords else 0.0,
        "runaway_samples": sum(s.get("num_tokens", 0) >= RUNAWAY_TOKENS for s in stats),
        "box_blocks": int(blocks),
        "fallback_rate": 100 * tot("switch_to_ar") / max(blocks, 1),
        "fallback_format_rate": 100 * tot("switch_format") / max(blocks, 1),
        "fallback_ambig_rate": 100 * tot("switch_ambig") / max(blocks, 1),
        "coord_top1": tot("sum_coord_top1") / max(coords, 1),
        "coord_entropy": tot("sum_coord_entropy") / max(coords, 1),
        "coord_low_conf_rate": 100 * tot("num_coords_low_conf") / max(coords, 1),
        # aggregate throughput: total boxes / total generation time
        "bps": tot("num_boxes") / max(tot("generate_time(s)"), 1e-9),
        "forward_steps_per_box": tot("forward_step") / max(tot("num_boxes"), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=path/to/answer.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--wandb_id", default=None, help="log rows to this (resumable) W&B run in project LocateAnything")
    ap.add_argument("--wandb_config", default="{}", help="JSON config attached to the W&B run")
    args = ap.parse_args()

    rows = []
    for spec in args.run:
        name, path = spec.split("=", 1)
        data = [json.loads(l) for l in open(path)]
        dec = decode_summary(data)
        for ds, f1 in f1_by_dataset(data).items():
            rows.append({"run": name, "dataset": ds, **f1, **dec})

    cols = ["run", "dataset", "f1_miou", "f1_50", "fallback_rate_macro", "fallback_ambig_rate_macro",
            "fallback_format_rate_macro", "coord_entropy_macro", "coord_top1_macro", "runaway_samples",
            "fallback_rate", "fallback_format_rate", "fallback_ambig_rate", "coord_entropy", "bps",
            "forward_steps_per_box", "samples"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r.get(c), float) else str(r.get(c, "-")) for c in cols) + " |")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)
    if args.wandb_id:
        log_wandb(rows, cols, args.wandb_id, json.loads(args.wandb_config))


def log_wandb(rows, cols, run_id, config):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from eaglevl.train.wandb_env import load_env

    load_env()
    import wandb

    run = wandb.init(project=os.environ["WANDB_PROJECT"], id=run_id, name=run_id, resume="allow",
                     job_type="eval", group=config.get("experiment"), config=config)
    for r in rows:
        for c in cols[2:]:
            if isinstance(r.get(c), (int, float)):
                run.summary[f"{r['dataset']}/{c}"] = r[c]
    run.log({"results": wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in rows])})
    run.finish()


if __name__ == "__main__":
    main()
