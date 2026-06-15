import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import normalize
from tqdm import tqdm

from common import ensure_dir, iter_images, label_from_path, load_config, resolve_project_path


def read_gray(path, image_size):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    h, w = image.shape[:2]
    scale = image_size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def create_detector(max_keypoints):
    if hasattr(cv2, "SIFT_create"):
        return "sift", cv2.SIFT_create(nfeatures=max_keypoints)
    return "orb", cv2.ORB_create(nfeatures=max_keypoints)


def extract_descriptors(path, detector, image_size, max_descriptors):
    gray = read_gray(path, image_size)
    _keypoints, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None or len(descriptors) == 0:
        return None
    descriptors = descriptors.astype(np.float32)
    if len(descriptors) > max_descriptors:
        descriptors = descriptors[:max_descriptors]
    return descriptors


def collect_dictionary_descriptors(paths, detector, image_size, max_descriptors, max_images, max_total_descriptors):
    collected = []
    failed = []
    used_images = 0

    for path in tqdm(paths[:max_images], desc="Collect SIFT descriptors", unit="img"):
        try:
            descriptors = extract_descriptors(path, detector, image_size, max_descriptors)
        except Exception as exc:
            failed.append((str(path), str(exc)))
            continue

        if descriptors is None:
            continue
        collected.append(descriptors)
        used_images += 1

        if sum(len(x) for x in collected) >= max_total_descriptors:
            break

    if not collected:
        raise RuntimeError("No descriptors collected for visual vocabulary.")

    descriptors = np.vstack(collected).astype(np.float32)
    if len(descriptors) > max_total_descriptors:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(descriptors), size=max_total_descriptors, replace=False)
        descriptors = descriptors[idx]

    return descriptors, used_images, failed


def fit_vocabulary(descriptors, vocab_size, batch_size):
    actual_vocab_size = min(vocab_size, len(descriptors))
    model = MiniBatchKMeans(
        n_clusters=actual_vocab_size,
        batch_size=batch_size,
        random_state=42,
        n_init=3,
        max_iter=120,
        reassignment_ratio=0.01,
        verbose=0,
    )
    model.fit(descriptors)
    return model


def encode_bovw(path, detector, kmeans, image_size, max_descriptors):
    descriptors = extract_descriptors(path, detector, image_size, max_descriptors)
    vocab_size = kmeans.cluster_centers_.shape[0]
    hist = np.zeros(vocab_size, dtype=np.float32)
    if descriptors is None:
        return hist

    words, _ = pairwise_distances_argmin_min(descriptors, kmeans.cluster_centers_)
    hist += np.bincount(words, minlength=vocab_size).astype(np.float32)
    hist = np.sqrt(hist)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def encode_images(paths, detector, kmeans, image_size, max_descriptors, title):
    features = []
    valid_paths = []
    failed = []

    for path in tqdm(paths, desc=title, unit="img"):
        try:
            features.append(encode_bovw(path, detector, kmeans, image_size, max_descriptors))
            valid_paths.append(path)
        except Exception as exc:
            failed.append((str(path), str(exc)))

    matrix = np.vstack(features).astype(np.float32)
    matrix = normalize(matrix, norm="l2", axis=1).astype(np.float32)
    return valid_paths, matrix, failed


def compute_topk(query_features, base_features, top_k):
    similarities = query_features @ base_features.T
    k = min(top_k, base_features.shape[0])
    candidate_idx = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
    candidate_scores = np.take_along_axis(similarities, candidate_idx, axis=1)
    order = np.argsort(-candidate_scores, axis=1)
    top_indices = np.take_along_axis(candidate_idx, order, axis=1)
    top_scores = np.take_along_axis(candidate_scores, order, axis=1)
    return top_indices, top_scores


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


def save_cache(output_dir, kmeans, base_paths, base_features, query_paths, query_features):
    np.savez_compressed(
        output_dir / "features_sift_bovw.npz",
        vocab_centers=kmeans.cluster_centers_.astype(np.float32),
        base_paths=np.array([str(p) for p in base_paths]),
        base_features=base_features,
        query_paths=np.array([str(p) for p in query_paths]),
        query_features=query_features,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default="outputs/retrieval_sift")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--max-keypoints", type=int, default=500)
    parser.add_argument("--max-descriptors-per-image", type=int, default=120)
    parser.add_argument("--dictionary-max-images", type=int, default=2000)
    parser.add_argument("--dictionary-max-descriptors", type=int, default=120000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-base", type=int, default=None, help="Optional debug limit.")
    parser.add_argument("--max-query", type=int, default=None, help="Optional debug limit.")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    top_k = max(int(k) for k in config["retrieval"]["top_k"])
    output_dir = ensure_dir(resolve_project_path(args.config, args.output_dir))

    base_dir = resolve_project_path(args.config, data_cfg["retrieval_base_dir"])
    query_dir = resolve_project_path(args.config, data_cfg["retrieval_query_dir"])
    base_paths = iter_images(base_dir)
    query_paths = iter_images(query_dir)
    if args.max_base:
        base_paths = base_paths[: args.max_base]
    if args.max_query:
        query_paths = query_paths[: args.max_query]

    detector_name, detector = create_detector(args.max_keypoints)

    print("=== Run SIFT-BoVW Retrieval ===")
    print(f"Detector: {detector_name}")
    print(f"Base images: {len(base_paths)}")
    print(f"Query images: {len(query_paths)}")
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Dictionary images: {min(args.dictionary_max_images, len(base_paths))}")
    print(f"Output dir: {output_dir}")

    descriptors, used_images, dict_failed = collect_dictionary_descriptors(
        base_paths,
        detector,
        args.image_size,
        args.max_descriptors_per_image,
        args.dictionary_max_images,
        args.dictionary_max_descriptors,
    )
    print(f"Collected descriptors: {descriptors.shape[0]} from {used_images} images")
    print("Fitting visual vocabulary...")
    kmeans = fit_vocabulary(descriptors, args.vocab_size, args.batch_size)

    base_paths, base_features, base_failed = encode_images(
        base_paths,
        detector,
        kmeans,
        args.image_size,
        args.max_descriptors_per_image,
        "Encode base BoVW",
    )
    query_paths, query_features, query_failed = encode_images(
        query_paths,
        detector,
        kmeans,
        args.image_size,
        args.max_descriptors_per_image,
        "Encode query BoVW",
    )

    print("Computing similarities...")
    top_indices, top_scores = compute_topk(query_features, base_features, top_k)
    results = build_results_dataframe(query_paths, base_paths, top_indices, top_scores)

    results_path = output_dir / "retrieval_results.csv"
    failures_path = output_dir / "feature_failures.csv"
    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    save_cache(output_dir, kmeans, base_paths, base_features, query_paths, query_features)

    failures = pd.DataFrame(dict_failed + base_failed + query_failed, columns=["path", "error"])
    failures.to_csv(failures_path, index=False, encoding="utf-8-sig")

    print(f"Saved results: {results_path}")
    print(f"Saved features: {output_dir / 'features_sift_bovw.npz'}")
    print(f"Saved failures: {failures_path} ({len(failures)} rows)")
    print("\nSample Top-5:")
    sample = results[results["query_name"] == results["query_name"].iloc[0]].head(5)
    print(sample[["query_name", "rank", "base_name", "score", "is_relevant"]].to_string(index=False))


if __name__ == "__main__":
    main()
