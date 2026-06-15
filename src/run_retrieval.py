import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from tqdm import tqdm

from common import ensure_dir, iter_images, label_from_path, load_config, resolve_project_path


def read_image_bgr(path, image_size):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)


def extract_feature(path, image_size=224, histogram_bins=8):
    image = read_image_bgr(path, image_size)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [histogram_bins, histogram_bins, histogram_bins],
        [0, 180, 0, 256, 0, 256],
    ).astype(np.float32)
    hist = cv2.normalize(hist, hist).flatten()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    thumb = thumb.flatten()
    thumb = (thumb - thumb.mean()) / (thumb.std() + 1e-6)

    feature = np.concatenate([hist, 0.35 * thumb]).astype(np.float32)
    return feature


def extract_features(paths, image_size, histogram_bins, title):
    features = []
    valid_paths = []
    failed = []

    for path in tqdm(paths, desc=title, unit="img"):
        try:
            features.append(extract_feature(path, image_size=image_size, histogram_bins=histogram_bins))
            valid_paths.append(path)
        except Exception as exc:
            failed.append((str(path), str(exc)))

    if not features:
        raise RuntimeError(f"No valid images found for {title}.")

    matrix = np.vstack(features).astype(np.float32)
    matrix = normalize(matrix, norm="l2", axis=1).astype(np.float32)
    return valid_paths, matrix, failed


def compute_topk(query_features, base_features, top_k):
    similarities = query_features @ base_features.T
    k = min(top_k, base_features.shape[0])
    # argpartition is much faster than full sort, then sort only selected candidates.
    candidate_idx = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
    candidate_scores = np.take_along_axis(similarities, candidate_idx, axis=1)
    order = np.argsort(-candidate_scores, axis=1)
    top_indices = np.take_along_axis(candidate_idx, order, axis=1)
    top_scores = np.take_along_axis(candidate_scores, order, axis=1)
    return top_indices, top_scores


def save_feature_cache(output_dir, base_paths, base_features, query_paths, query_features):
    np.savez_compressed(
        output_dir / "features_color_struct.npz",
        base_paths=np.array([str(p) for p in base_paths]),
        base_features=base_features,
        query_paths=np.array([str(p) for p in query_paths]),
        query_features=query_features,
    )


def build_results_dataframe(query_paths, base_paths, top_indices, top_scores):
    rows = []
    for query_i, query_path in enumerate(query_paths):
        query_label = label_from_path(query_path)
        for rank, (base_i, score) in enumerate(zip(top_indices[query_i], top_scores[query_i]), start=1):
            base_path = base_paths[int(base_i)]
            base_label = label_from_path(base_path)
            rows.append(
                {
                    "query_path": str(query_path),
                    "query_name": query_path.name,
                    "query_label": query_label,
                    "rank": rank,
                    "base_path": str(base_path),
                    "base_name": base_path.name,
                    "base_label": base_label,
                    "score": float(score),
                    "is_relevant": int(query_label == base_label),
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--max-base", type=int, default=None, help="Optional debug limit.")
    parser.add_argument("--max-query", type=int, default=None, help="Optional debug limit.")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    retrieval_cfg = config["retrieval"]
    output_dir = ensure_dir(resolve_project_path(args.config, config["outputs"]["retrieval_dir"]))

    base_dir = resolve_project_path(args.config, data_cfg["retrieval_base_dir"])
    query_dir = resolve_project_path(args.config, data_cfg["retrieval_query_dir"])
    image_size = int(retrieval_cfg["image_size"])
    histogram_bins = int(retrieval_cfg["histogram_bins"])
    top_k = max(int(k) for k in retrieval_cfg["top_k"])

    base_paths = iter_images(base_dir)
    query_paths = iter_images(query_dir)
    if args.max_base:
        base_paths = base_paths[: args.max_base]
    if args.max_query:
        query_paths = query_paths[: args.max_query]

    print("=== Run Retrieval Baseline ===")
    print(f"Base dir: {base_dir}")
    print(f"Query dir: {query_dir}")
    print(f"Base images: {len(base_paths)}")
    print(f"Query images: {len(query_paths)}")
    print(f"Feature: HSV histogram bins={histogram_bins} + gray thumbnail, image_size={image_size}")
    print(f"TopK: {top_k}")

    base_paths, base_features, base_failed = extract_features(
        base_paths, image_size=image_size, histogram_bins=histogram_bins, title="Base features"
    )
    query_paths, query_features, query_failed = extract_features(
        query_paths, image_size=image_size, histogram_bins=histogram_bins, title="Query features"
    )

    print("Computing similarities...")
    top_indices, top_scores = compute_topk(query_features, base_features, top_k)
    results = build_results_dataframe(query_paths, base_paths, top_indices, top_scores)

    results_path = output_dir / "retrieval_results.csv"
    failures_path = output_dir / "feature_failures.csv"
    features_path = output_dir / "features_color_struct.npz"

    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    save_feature_cache(output_dir, base_paths, base_features, query_paths, query_features)

    failures = pd.DataFrame(base_failed + query_failed, columns=["path", "error"])
    failures.to_csv(failures_path, index=False, encoding="utf-8-sig")

    print(f"Saved results: {results_path}")
    print(f"Saved features: {features_path}")
    print(f"Saved failures: {failures_path} ({len(failures)} rows)")
    print("\nSample Top-5:")
    sample = results[results["query_name"] == results["query_name"].iloc[0]].head(5)
    print(sample[["query_name", "rank", "base_name", "score", "is_relevant"]].to_string(index=False))


if __name__ == "__main__":
    main()
