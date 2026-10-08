"""Log the top-k coordinate candidates at every hybrid-mode ambiguity trigger, for a chosen set of samples.

Used to diagnose *why* a model falls back: a trigger needs top-1 < 0.9, >1 coordinate token in the top-k and a
top-k spread > 60. The spread can come from probability mass split across distant modes (two candidate objects)
or from a diffuse distribution; the logged candidates distinguish the two.

Usage: python diagnose_ambiguity.py --model_path ... [--lora_path ...] --samples flips.jsonl --out diag.jsonl
"""
import argparse
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference_grounding_ddp import LocateAnythingWorker, normalize_categories  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lora_path", default=None)
    ap.add_argument("--samples", required=True, help="jsonl rows in eval format")
    ap.add_argument("--image_root", default="work_dirs/evaldata/images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    worker = LocateAnythingWorker(args.model_path, device=args.device, generation_mode="hybrid", dtype=args.dtype,
                                  lora_path=args.lora_path, seed=0)
    gu = sys.modules[type(worker.model).__module__.rsplit(".", 1)[0] + ".generate_utils"]
    events = []
    orig = gu.coord_is_abnormal

    def logged(pos_probs, pos_ids, token_ids):
        out = orig(pos_probs, pos_ids, token_ids)
        c0 = token_ids["coord_start_token_id"]
        for i in out.nonzero().flatten().tolist():
            events[-1]["triggers"].append({
                "slot": i,
                "topk": [(int(t) - c0 if c0 <= int(t) <= token_ids["coord_end_token_id"] else f"tok{int(t)}",
                          round(float(p), 4)) for p, t in zip(pos_probs[i], pos_ids[i])],
            })
        return out

    gu.coord_is_abnormal = logged
    with open(args.out, "w") as f:
        for row in map(json.loads, open(args.samples)):
            events.append({"image_path": row["image_path"], "triggers": []})
            image = Image.open(os.path.join(args.image_root, row["image_path"])).convert("RGB")
            q = None
            if row["dataset_name"] in ("RefCOCOg_test", "RefCOCOg_val"):
                q = "Locate a single instance that matches the following description: "
            out, _ = worker.generate(image, normalize_categories(row["categories"]), max_new_tokens=512,
                                     question_override=q)
            events[-1]["response"] = out
            f.write(json.dumps(events[-1]) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
