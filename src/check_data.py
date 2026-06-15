import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_simple_yaml(path):
    """Load the small project config without requiring PyYAML."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    root = {}
    stack = [(0, root)]
    pending_list_key = None

    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()

        while stack and indent < stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if line.startswith("- "):
            if pending_list_key is None:
                raise ValueError(f"List item without key: {raw}")
            current[pending_list_key].append(parse_scalar(line[2:]))
            continue

        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()

        if value == "":
            next_obj = {}
            current[key] = next_obj
            stack.append((indent + 2, next_obj))
            pending_list_key = None
        else:
            if value == "[]":
                current[key] = []
                pending_list_key = key
            else:
                current[key] = parse_scalar(value)
                pending_list_key = None

        if key in {"landmark_prefixes", "top_k"} and value == "":
            current[key] = []
            stack.append((indent + 2, current))
            pending_list_key = key

    return root


def parse_scalar(value):
    value = value.strip().strip('"').strip("'")
    if value.isdigit():
        return int(value)
    return value


def project_path(config_path, raw_path):
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (Path(config_path).resolve().parent.parent / p).resolve()


def iter_images(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def label_from_name(path):
    return path.stem.split("-", 1)[0].lower()


def summarize_images(paths, landmark_prefixes):
    prefixes = set(landmark_prefixes)
    counts = Counter(label_from_name(p) for p in paths)
    return {prefix: counts.get(prefix, 0) for prefix in landmark_prefixes}, counts


def inspect_labelme_detection(data_dir):
    data_dir = Path(data_dir)
    json_files = sorted(data_dir.glob("*.json"))
    image_files = iter_images(data_dir)
    image_by_stem = {p.stem: p for p in image_files}

    missing_images = []
    shape_labels = Counter()
    shape_types = Counter()
    files_by_prefix = Counter()
    json_errors = []

    for json_path in json_files:
        files_by_prefix[label_from_name(json_path)] += 1
        try:
            text = json_path.read_text(encoding="utf-8")
            text = re.sub(r'\s*"imageData"\s*:\s*".*?",\s*', "", text, flags=re.S)
            data = json.loads(text)
        except Exception as exc:
            json_errors.append((json_path.name, str(exc)))
            continue

        image_name = data.get("imagePath")
        stem = Path(image_name).stem if image_name else json_path.stem
        if stem not in image_by_stem:
            missing_images.append(json_path.name)

        for shape in data.get("shapes", []):
            shape_labels[str(shape.get("label", ""))] += 1
            shape_types[str(shape.get("shape_type", ""))] += 1

    image_stems = {p.stem for p in image_files}
    json_stems = {p.stem for p in json_files}
    missing_json = sorted(image_by_stem[s].name for s in image_stems - json_stems if s in image_by_stem)

    return {
        "json_count": len(json_files),
        "image_count": len(image_files),
        "missing_images": missing_images[:20],
        "missing_image_count": len(missing_images),
        "missing_json": missing_json[:20],
        "missing_json_count": len(missing_json),
        "shape_labels": shape_labels,
        "shape_types": shape_types,
        "files_by_prefix": files_by_prefix,
        "json_errors": json_errors[:20],
        "json_error_count": len(json_errors),
    }


def maybe_check_image_open(paths, limit=10):
    try:
        from PIL import Image
    except Exception:
        return "Pillow is not available; skipped image open check."

    bad = []
    for path in paths[:limit]:
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as exc:
            bad.append((str(path), str(exc)))
    if bad:
        return f"Image open check found {len(bad)} bad files in first {limit}: {bad[:3]}"
    return f"Image open check passed for first {min(limit, len(paths))} files."


def print_counter(title, counter, limit=20):
    print(f"\n{title}")
    for key, value in counter.most_common(limit):
        print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    data_cfg = config["data"]
    landmark_prefixes = data_cfg["landmark_prefixes"]

    base_dir = project_path(args.config, data_cfg["retrieval_base_dir"])
    query_dir = project_path(args.config, data_cfg["retrieval_query_dir"])
    detection_dir = project_path(args.config, data_cfg["object_detection_dir"])

    base_images = iter_images(base_dir)
    query_images = iter_images(query_dir)
    base_expected, base_all = summarize_images(base_images, landmark_prefixes)
    query_expected, query_all = summarize_images(query_images, landmark_prefixes)
    detection = inspect_labelme_detection(detection_dir)

    print("=== Lab 2 Data Check ===")
    print(f"Config: {Path(args.config).resolve()}")
    print(f"Retrieval base dir: {base_dir}")
    print(f"Retrieval query dir: {query_dir}")
    print(f"Detection data dir: {detection_dir}")

    print("\n[Image Retrieval]")
    print(f"Base images: {len(base_images)}")
    print(f"Query images: {len(query_images)}")
    print(maybe_check_image_open(base_images + query_images, limit=10))

    print_counter("Base landmark distribution", Counter(base_expected))
    print_counter("Query landmark distribution", Counter(query_expected))

    unknown_base = {k: v for k, v in base_all.items() if k not in set(landmark_prefixes)}
    if unknown_base:
        print_counter("Base non-landmark/unknown prefixes", Counter(unknown_base), limit=10)

    print("\n[Object Detection]")
    print(f"Detection images: {detection['image_count']}")
    print(f"Detection json files: {detection['json_count']}")
    print(f"Json without matching image: {detection['missing_image_count']}")
    print(f"Images without matching json: {detection['missing_json_count']}")
    if detection["missing_images"]:
        print(f"  Examples: {', '.join(detection['missing_images'][:5])}")
    if detection["missing_json"]:
        print(f"  Examples: {', '.join(detection['missing_json'][:5])}")
    if detection["json_error_count"]:
        print(f"Json parse errors: {detection['json_error_count']}")
        print(f"  Examples: {detection['json_errors'][:3]}")

    print_counter("Detection annotation labels", detection["shape_labels"], limit=20)
    print_counter("Detection shape types", detection["shape_types"], limit=10)
    print_counter("Detection files by landmark prefix", detection["files_by_prefix"], limit=20)

    print("\nNext step: implement retrieval baseline in src/run_retrieval.py.")


if __name__ == "__main__":
    main()
