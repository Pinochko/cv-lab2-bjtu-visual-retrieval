import argparse
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

from common import ensure_dir, load_config, resolve_project_path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_font(size=20):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def read_labelme_json(path):
    text = Path(path).read_text(encoding="utf-8")
    text = re.sub(r'\s*"imageData"\s*:\s*".*?",\s*', "", text, flags=re.S)
    return json.loads(text)


def find_image(data_dir, stem):
    for ext in IMAGE_EXTS:
        candidate = data_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    for path in data_dir.glob(stem + ".*"):
        if path.suffix.lower() in IMAGE_EXTS:
            return path
    return None


def shape_bbox(shape):
    points = shape.get("points", [])
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def draw_detection(image_path, json_path, output_path):
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    draw = ImageDraw.Draw(image)
    font = load_font(max(16, image.width // 45))
    data = read_labelme_json(json_path)

    for shape in data.get("shapes", []):
        bbox = shape_bbox(shape)
        if bbox is None:
            continue
        label = str(shape.get("label", "text"))
        x1, y1, x2, y2 = bbox
        width = max(3, image.width // 220)
        color = (40, 150, 80)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = int(x1)
        text_y = max(0, int(y1) - text_h - 8)
        draw.rectangle([text_x, text_y, text_x + text_w + 10, text_y + text_h + 8], fill=color)
        draw.text((text_x + 5, text_y + 4), label, fill="white", font=font)

    ensure_dir(output_path.parent)
    image.save(output_path, quality=92)
    return len(data.get("shapes", []))


def resize_to_height(image, target_h):
    ratio = target_h / image.height
    target_w = max(1, int(image.width * ratio))
    return image.resize((target_w, target_h), Image.Resampling.LANCZOS)


def combine_retrieval_detection(retrieval_path, detection_path, output_path):
    retrieval = Image.open(retrieval_path).convert("RGB")
    detection = Image.open(detection_path).convert("RGB")
    target_h = retrieval.height
    detection = resize_to_height(detection, target_h)
    gap = 14
    bg = (245, 247, 250)
    canvas = Image.new("RGB", (retrieval.width + gap + detection.width, target_h), bg)
    canvas.paste(retrieval, (0, 0))
    canvas.paste(detection, (retrieval.width + gap, 0))
    ensure_dir(output_path.parent)
    canvas.save(output_path, quality=92)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--manifest", default="outputs/demo_cases_resnet50/manifest.csv")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/detection_resnet50")
    parser.add_argument("--combined-dir", default="outputs/demo_retrieval_detection_resnet50")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = resolve_project_path(args.config, args.manifest)
    data_dir = resolve_project_path(args.config, args.data_dir or config["data"]["object_detection_dir"])
    output_dir = ensure_dir(resolve_project_path(args.config, args.output_dir))
    combined_dir = ensure_dir(resolve_project_path(args.config, args.combined_dir))

    manifest = pd.read_csv(manifest_path)
    rows = []

    print("=== Visualize Detection Boxes ===")
    print(f"Manifest: {manifest_path}")
    print(f"Detection data: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Combined dir: {combined_dir}")

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Draw detections", unit="case"):
        query_name = row["query_name"]
        stem = Path(query_name).stem
        image_path = find_image(data_dir, stem)
        json_path = data_dir / f"{stem}.json"
        if image_path is None or not json_path.exists():
            rows.append({**row.to_dict(), "status": "missing", "detection_path": "", "combined_path": "", "num_shapes": 0})
            continue

        label = row["label"]
        case_index = int(row["case_index"])
        detection_path = output_dir / f"{label}_case_{case_index}_{stem}_detection.jpg"
        num_shapes = draw_detection(image_path, json_path, detection_path)

        retrieval_path = Path(row["output_path"])
        combined_path = combined_dir / f"{label}_case_{case_index}_{stem}_retrieval_detection.jpg"
        if retrieval_path.exists():
            combine_retrieval_detection(retrieval_path, detection_path, combined_path)
            status = "ok"
        else:
            status = "detection_only"

        rows.append(
            {
                **row.to_dict(),
                "status": status,
                "image_path": str(image_path),
                "json_path": str(json_path),
                "detection_path": str(detection_path),
                "combined_path": str(combined_path),
                "num_shapes": num_shapes,
            }
        )

    out_manifest = pd.DataFrame(rows)
    out_manifest_path = combined_dir / "manifest.csv"
    out_manifest.to_csv(out_manifest_path, index=False, encoding="utf-8-sig")
    print(f"Saved manifest: {out_manifest_path}")
    print(f"OK cases: {(out_manifest['status'] == 'ok').sum()} / {len(out_manifest)}")


if __name__ == "__main__":
    main()
