import argparse
import shutil
from pathlib import Path

import pandas as pd

from common import ensure_dir, resolve_project_path


def copy_file(src, dst):
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def copy_glob(src_dir, pattern, dst_dir):
    copied = []
    for src in sorted(Path(src_dir).glob(pattern)):
        dst = Path(dst_dir) / src.name
        if copy_file(src, dst):
            copied.append(dst)
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default="outputs/report_assets")
    args = parser.parse_args()

    out_root = ensure_dir(resolve_project_path(args.config, args.output_dir))
    tables_dir = ensure_dir(out_root / "tables")
    figures_dir = ensure_dir(out_root / "figures")
    cases_dir = ensure_dir(out_root / "cases_retrieval_detection")
    retrieval_cases_dir = ensure_dir(out_root / "cases_retrieval_only")
    detection_dir = ensure_dir(out_root / "cases_detection_only")

    items = []

    files_to_copy = [
        ("table", "method_comparison_overall.csv", "outputs/comparison/method_comparison_overall.csv", tables_dir),
        ("table", "method_comparison_by_landmark.csv", "outputs/comparison/method_comparison_by_landmark.csv", tables_dir),
        ("table", "resnet50_precision_at_k.csv", "outputs/retrieval_resnet50/precision_at_k.csv", tables_dir),
        ("table", "color_precision_at_k.csv", "outputs/retrieval/precision_at_k.csv", tables_dir),
        ("table", "sift_precision_at_k.csv", "outputs/retrieval_sift/precision_at_k.csv", tables_dir),
        ("figure", "method_comparison_overall.png", "outputs/comparison/method_comparison_overall.png", figures_dir),
        ("figure", "method_comparison_p20_by_landmark.png", "outputs/comparison/method_comparison_p20_by_landmark.png", figures_dir),
        ("figure", "resnet50_p_at_k_all_landmarks.png", "outputs/figures_resnet50/p_at_k_all_landmarks.png", figures_dir),
    ]

    for kind, name, src, dst_dir in files_to_copy:
        src_path = resolve_project_path(args.config, src)
        dst_path = dst_dir / name
        ok = copy_file(src_path, dst_path)
        items.append({"kind": kind, "name": name, "source": str(src_path), "path": str(dst_path), "status": "ok" if ok else "missing"})

    for dst in copy_glob(resolve_project_path(args.config, "outputs/figures_resnet50"), "p_at_k_*.png", figures_dir / "resnet50_landmark_curves"):
        items.append({"kind": "figure", "name": dst.name, "source": "outputs/figures_resnet50", "path": str(dst), "status": "ok"})

    for dst in copy_glob(resolve_project_path(args.config, "outputs/demo_retrieval_detection_resnet50"), "*.jpg", cases_dir):
        items.append({"kind": "case_retrieval_detection", "name": dst.name, "source": "outputs/demo_retrieval_detection_resnet50", "path": str(dst), "status": "ok"})

    for dst in copy_glob(resolve_project_path(args.config, "outputs/demo_cases_resnet50"), "*.jpg", retrieval_cases_dir):
        items.append({"kind": "case_retrieval", "name": dst.name, "source": "outputs/demo_cases_resnet50", "path": str(dst), "status": "ok"})

    for dst in copy_glob(resolve_project_path(args.config, "outputs/detection_resnet50"), "*.jpg", detection_dir):
        items.append({"kind": "case_detection", "name": dst.name, "source": "outputs/detection_resnet50", "path": str(dst), "status": "ok"})

    # Copy manifests last.
    manifest_sources = [
        ("manifest_retrieval_detection.csv", "outputs/demo_retrieval_detection_resnet50/manifest.csv"),
        ("manifest_retrieval.csv", "outputs/demo_cases_resnet50/manifest.csv"),
    ]
    for name, src in manifest_sources:
        src_path = resolve_project_path(args.config, src)
        dst_path = tables_dir / name
        ok = copy_file(src_path, dst_path)
        items.append({"kind": "manifest", "name": name, "source": str(src_path), "path": str(dst_path), "status": "ok" if ok else "missing"})

    asset_manifest = pd.DataFrame(items)
    manifest_path = out_root / "asset_manifest.csv"
    asset_manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    summary = asset_manifest.groupby(["kind", "status"]).size().reset_index(name="count")
    summary_path = out_root / "asset_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("=== Prepare Report Assets ===")
    print(f"Output: {out_root}")
    print(f"Asset manifest: {manifest_path}")
    print(f"Asset summary: {summary_path}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
