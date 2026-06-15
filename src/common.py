from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root_from_config(config_path):
    return Path(config_path).resolve().parent.parent


def resolve_project_path(config_path, raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (project_root_from_config(config_path) / path).resolve()


def iter_images(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def label_from_path(path):
    return Path(path).stem.split("-", 1)[0].lower()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)
