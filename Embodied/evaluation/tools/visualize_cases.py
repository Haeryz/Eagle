"""Side-by-side qualitative panels for slides: GT | baseline | method, boxes coloured by error type.

Per dataset it picks, from saved predictions on the same images, the largest per-image F1@0.5 improvements and
regressions of the method over the baseline, plus the images with the most baseline NTP fallbacks. Each panel title
shows F1@0.5 and the number of NTP fallbacks for that image.

Colours: correct = green, localization = orange, duplicate = magenta, class confusion = yellow,
background false positive = red, missed GT (in the GT panel) = dashed cyan, other GT = white.

Usage: python visualize_cases.py --dataset COCO --base E0=.../answer.jsonl --method E3=.../answer.jsonl \
           --image_root work_dirs/evaldata/images --out_dir work_dirs/report_assets --k 3
"""
import argparse
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from error_analysis import OCR_DATASETS, analyze_sample  # noqa: E402

COLORS = {"correct": (40, 200, 70), "localize": (255, 150, 0), "duplicate": (230, 0, 230),
          "class": (255, 230, 0), "background": (240, 30, 30)}
PANEL_W = 640


def load(path):
    return {d["image_path"] + d.get("question", ""): d for d in map(json.loads, open(path))}


def per_image(d):
    gt = d["gt"] if isinstance(d["gt"], dict) else {}
    pred = d.get("extracted_predictions", {})
    if d.get("dataset_name") in OCR_DATASETS:
        gt = {"text": [b for v in gt.values() for b in v]}
        pred = {"text": [b for v in pred.values() for b in v]}
    counts, labels, missed = analyze_sample(pred, gt, return_labels=True)
    n_pred = len(labels)
    p = counts["correct"] / n_pred if n_pred else 0.0
    r = counts["correct"] / counts["gt"] if counts["gt"] else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return f1, labels, missed, gt, int(d.get("decode_stats", {}).get("switch_to_ar", 0))


def font(size):
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def dashed_rect(draw, box, color, width, dash=8):
    x1, y1, x2, y2 = box
    for (a, b), (c, d) in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
        n = max(int(max(abs(c - a), abs(d - b)) / dash), 1)
        for i in range(0, n, 2):
            t0, t1 = i / n, min((i + 1) / n, 1)
            draw.line([(a + (c - a) * t0, b + (d - b) * t0), (a + (c - a) * t1, b + (d - b) * t1)], fill=color, width=width)


def panel(image, title, boxes, scale):
    im = image.copy().resize((PANEL_W, int(image.height * scale)))
    draw = ImageDraw.Draw(im)
    lw = 3 if len(boxes) < 60 else 2
    for box, color, dashed in boxes:
        b = [v * scale for v in box]
        if dashed:
            dashed_rect(draw, b, color, lw)
        else:
            draw.rectangle(b, outline=color, width=lw)
    bar = Image.new("RGB", (PANEL_W, 34), (25, 25, 25))
    ImageDraw.Draw(bar).text((8, 6), title, fill=(255, 255, 255), font=font(18))
    out = Image.new("RGB", (PANEL_W, im.height + 34), (25, 25, 25))
    out.paste(bar, (0, 0))
    out.paste(im, (0, 34))
    return out


def render(key, base, method, names, image_root, out_path):
    db, dm = base[key], method[key]
    image = Image.open(os.path.join(image_root, db["image_path"])).convert("RGB")
    scale = PANEL_W / image.width
    f1b, lb, missed_b, gt, fb_b = per_image(db)
    f1m, lm, missed_m, _, fb_m = per_image(dm)
    missed_ids = {tuple(b) for _, b in missed_m}
    gt_boxes = [(b, (0, 220, 255) if tuple(b) in missed_ids else (255, 255, 255), tuple(b) in missed_ids)
                for v in gt.values() for b in v]
    panels = [
        panel(image, f"Ground truth ({len(gt_boxes)} boxes; dashed = missed by {names[1]})", gt_boxes, scale),
        panel(image, f"{names[0]}: F1@0.5 {100 * f1b:.0f} | fallbacks {fb_b}", [(b, COLORS[l], False) for _, b, l in lb], scale),
        panel(image, f"{names[1]}: F1@0.5 {100 * f1m:.0f} | fallbacks {fb_m}", [(b, COLORS[l], False) for _, b, l in lm], scale),
    ]
    h = max(p.height for p in panels)
    q = db.get("question", "")
    canvas = Image.new("RGB", (PANEL_W * 3 + 20, h + 64), (255, 255, 255))
    for i, p in enumerate(panels):
        canvas.paste(p, (i * (PANEL_W + 10), 0))
    d = ImageDraw.Draw(canvas)
    d.text((8, h + 6), (q[:170] + "…") if len(q) > 170 else q, fill=(0, 0, 0), font=font(16))
    x = 8
    for name, c in [("correct", COLORS["correct"]), ("localization", COLORS["localize"]),
                    ("duplicate", COLORS["duplicate"]), ("class confusion", COLORS["class"]),
                    ("background FP", COLORS["background"])]:
        d.rectangle([x, h + 36, x + 18, h + 54], fill=c)
        d.text((x + 24, h + 36), name, fill=(0, 0, 0), font=font(15))
        x += 190
    canvas.save(out_path, optimize=True)
    return f1b, f1m, fb_b, fb_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--image_root", default="work_dirs/evaldata/images")
    ap.add_argument("--out_dir", default="work_dirs/report_assets")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    (bn, bp), (mn, mp) = args.base.split("=", 1), args.method.split("=", 1)
    base, method = load(bp), load(mp)
    keys = [k for k in base if k in method and base[k].get("decode_stats", {}).get("num_tokens", 0) < 8000
            and method[k].get("decode_stats", {}).get("num_tokens", 0) < 8000]
    scored = {k: (per_image(base[k])[0], per_image(method[k])[0], per_image(base[k])[4]) for k in keys}
    picks = {
        "win": sorted(keys, key=lambda k: scored[k][0] - scored[k][1])[:args.k],
        "regression": sorted(keys, key=lambda k: scored[k][1] - scored[k][0])[:args.k],
        "most_fallbacks": sorted(keys, key=lambda k: -scored[k][2])[:args.k],
    }
    os.makedirs(args.out_dir, exist_ok=True)
    index = []
    for kind, ks in picks.items():
        for i, k in enumerate(ks):
            out = os.path.join(args.out_dir, f"{args.dataset}_{kind}_{i}.png")
            f1b, f1m, fb_b, fb_m = render(k, base, method, (bn, mn), args.image_root, out)
            index.append({"file": out, "kind": kind, "image": base[k]["image_path"], f"{bn}_f1_50": f1b,
                          f"{mn}_f1_50": f1m, f"{bn}_fallbacks": fb_b, f"{mn}_fallbacks": fb_m})
            print(f"{kind:15s} {out}  {bn} F1 {100 * f1b:.0f} fb {fb_b} -> {mn} F1 {100 * f1m:.0f} fb {fb_m}")
    with open(os.path.join(args.out_dir, f"{args.dataset}_index.json"), "w") as f:
        json.dump(index, f, indent=2)


if __name__ == "__main__":
    main()
