import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import ensure_dir, resolve_project_path


def read_summary(path, method):
    df = pd.read_csv(path)
    df.insert(0, "method", method)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default="outputs/comparison")
    args = parser.parse_args()

    output_dir = ensure_dir(resolve_project_path(args.config, args.output_dir))
    methods = [
        ("Color+Structure", resolve_project_path(args.config, "outputs/retrieval/precision_at_k.csv")),
        ("SIFT-BoVW", resolve_project_path(args.config, "outputs/retrieval_sift/precision_at_k.csv")),
        ("ResNet50", resolve_project_path(args.config, "outputs/retrieval_resnet50/precision_at_k.csv")),
    ]

    frames = []
    for method, path in methods:
        if not path.exists():
            raise FileNotFoundError(f"Missing metric file for {method}: {path}")
        frames.append(read_summary(path, method))

    comparison = pd.concat(frames, ignore_index=True)
    overall = comparison[comparison["scope"] == "overall"].copy()
    per_landmark = comparison[comparison["scope"] == "landmark"].copy()

    comparison_path = output_dir / "method_comparison_all.csv"
    overall_path = output_dir / "method_comparison_overall.csv"
    landmark_path = output_dir / "method_comparison_by_landmark.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    overall.to_csv(overall_path, index=False, encoding="utf-8-sig")
    per_landmark.to_csv(landmark_path, index=False, encoding="utf-8-sig")

    metric_cols = [c for c in overall.columns if c.startswith("P@")]
    plt.figure(figsize=(7.5, 4.8))
    x = range(len(overall))
    width = 0.22
    for i, metric in enumerate(metric_cols):
        offsets = [v + (i - 1) * width for v in x]
        plt.bar(offsets, overall[metric], width=width, label=metric)
    plt.xticks(list(x), overall["method"].tolist())
    plt.ylim(0, 1)
    plt.ylabel("Precision")
    plt.title("Retrieval Method Comparison")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "method_comparison_overall.png", dpi=180)
    plt.close()

    pivot = per_landmark.pivot(index="label", columns="method", values="P@20")
    pivot = pivot[[m[0] for m in methods]]
    pivot.plot(kind="bar", figsize=(11, 5.5), ylim=(0, 1), grid=True)
    plt.ylabel("P@20")
    plt.xlabel("Landmark")
    plt.title("P@20 by Landmark and Method")
    plt.tight_layout()
    plt.savefig(output_dir / "method_comparison_p20_by_landmark.png", dpi=180)
    plt.close()

    print("=== Method Comparison ===")
    print(f"Saved: {overall_path}")
    print(f"Saved: {landmark_path}")
    print(f"Saved figures: {output_dir}")
    print("\nOverall:")
    print(overall[["method", "num_queries"] + metric_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
