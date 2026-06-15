import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import ensure_dir, load_config, resolve_project_path


def precision_at_k(group, k):
    top = group.sort_values("rank").head(k)
    if top.empty:
        return 0.0
    return float(top["is_relevant"].mean())


def compute_metrics(results, top_k_values, landmark_prefixes):
    query_groups = list(results.groupby("query_name", sort=True))
    rows = []

    for query_name, group in query_groups:
        query_label = str(group["query_label"].iloc[0]).lower()
        row = {"scope": "query", "label": query_label, "query_name": query_name, "num_queries": 1}
        for k in top_k_values:
            row[f"P@{k}"] = precision_at_k(group, k)
        rows.append(row)

    query_metrics = pd.DataFrame(rows)

    per_class_rows = []
    for label in landmark_prefixes:
        subset = query_metrics[query_metrics["label"] == label]
        row = {"scope": "landmark", "label": label, "num_queries": int(len(subset))}
        for k in top_k_values:
            row[f"P@{k}"] = float(subset[f"P@{k}"].mean()) if not subset.empty else 0.0
        per_class_rows.append(row)

    class_metrics = pd.DataFrame(per_class_rows)
    overall = {"scope": "overall", "label": "all", "num_queries": int(len(query_metrics))}
    for k in top_k_values:
        overall[f"P@{k}"] = float(query_metrics[f"P@{k}"].mean())

    summary = pd.concat([class_metrics, pd.DataFrame([overall])], ignore_index=True)
    return query_metrics, summary


def plot_landmark_curves(summary, top_k_values, figures_dir):
    class_summary = summary[summary["scope"] == "landmark"].copy()
    for _, row in class_summary.iterrows():
        label = row["label"]
        y = [row[f"P@{k}"] for k in top_k_values]
        plt.figure(figsize=(5.5, 4))
        plt.plot(top_k_values, y, marker="o", linewidth=2)
        plt.ylim(0, 1)
        plt.xticks(top_k_values)
        plt.xlabel("Top-K")
        plt.ylabel("Precision")
        plt.title(f"Precision@K - {label}")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.tight_layout()
        plt.savefig(figures_dir / f"p_at_k_{label}.png", dpi=160)
        plt.close()


def plot_overall_bar(summary, top_k_values, figures_dir):
    class_summary = summary[summary["scope"] == "landmark"].copy()
    labels = class_summary["label"].tolist()

    plt.figure(figsize=(10, 5.5))
    width = 0.24
    x = range(len(labels))
    for i, k in enumerate(top_k_values):
        values = class_summary[f"P@{k}"].tolist()
        offsets = [v + (i - 1) * width for v in x]
        plt.bar(offsets, values, width=width, label=f"P@{k}")

    plt.ylim(0, 1)
    plt.xticks(list(x), labels)
    plt.xlabel("Landmark")
    plt.ylabel("Precision")
    plt.title("Precision@K by Landmark")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "p_at_k_all_landmarks.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--results", default=None, help="Optional retrieval_results.csv path.")
    parser.add_argument("--output-dir", default=None, help="Directory for metric CSV outputs.")
    parser.add_argument("--figures-dir", default=None, help="Directory for Precision@K figures.")
    args = parser.parse_args()

    config = load_config(args.config)
    top_k_values = [int(k) for k in config["retrieval"]["top_k"]]
    landmark_prefixes = [str(v).lower() for v in config["data"]["landmark_prefixes"]]

    retrieval_dir = ensure_dir(
        resolve_project_path(args.config, args.output_dir or config["outputs"]["retrieval_dir"])
    )
    figures_dir = ensure_dir(
        resolve_project_path(args.config, args.figures_dir or config["outputs"]["figures_dir"])
    )
    results_path = Path(args.results) if args.results else retrieval_dir / "retrieval_results.csv"

    if not results_path.exists():
        raise FileNotFoundError(f"Missing retrieval results: {results_path}")

    results = pd.read_csv(results_path)
    required = {"query_name", "query_label", "rank", "is_relevant"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Missing required columns in results: {sorted(missing)}")

    query_metrics, summary = compute_metrics(results, top_k_values, landmark_prefixes)

    query_metrics_path = retrieval_dir / "precision_at_k_by_query.csv"
    summary_path = retrieval_dir / "precision_at_k.csv"
    query_metrics.to_csv(query_metrics_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    plot_landmark_curves(summary, top_k_values, figures_dir)
    plot_overall_bar(summary, top_k_values, figures_dir)

    print("=== Retrieval Evaluation ===")
    print(f"Results: {results_path}")
    print(f"Saved query metrics: {query_metrics_path}")
    print(f"Saved summary metrics: {summary_path}")
    print(f"Saved figures: {figures_dir}")
    print("\nSummary:")
    cols = ["scope", "label", "num_queries"] + [f"P@{k}" for k in top_k_values]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
