"""Build fixed, seeded eval subsets (used for the fallback-reduction experiments on a single GPU)
and the list of COCO image ids that any eval set touches (excluded from self-distillation data)."""
import argparse
import json
import os
import random
import re

SIZES = {"COCO": 500, "LVIS": 500, "RefCOCOg_val": 500, "RefCOCOg_test": 500, "Dense200": None, "SROIE": None}


def coco_id(image_path):
    m = re.search(r"(\d{12})\.jpg$", image_path)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_dir", required=True, help="dir with box_eval/<DS>.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    excluded, needed = set(), set()
    for name, size in SIZES.items():
        lines = open(os.path.join(args.ann_dir, f"{name}.jsonl")).readlines()
        for line in lines:  # every eval image is excluded from training, not only the subset
            cid = coco_id(json.loads(line)["image_path"])
            if cid is not None:
                excluded.add(cid)
        if size is not None and size < len(lines):
            lines = random.Random(args.seed).sample(lines, size)
        with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w") as f:
            f.writelines(lines)
        needed.update(json.loads(l)["image_path"] for l in lines)
        print(f"{name}: {len(lines)} samples")

    with open(os.path.join(args.out_dir, "excluded_coco_ids.json"), "w") as f:
        json.dump(sorted(excluded), f)
    with open(os.path.join(args.out_dir, "needed_images.txt"), "w") as f:
        f.write("\n".join(sorted(needed)) + "\n")
    print(f"excluded coco ids: {len(excluded)}, images needed: {len(needed)}")


if __name__ == "__main__":
    main()
