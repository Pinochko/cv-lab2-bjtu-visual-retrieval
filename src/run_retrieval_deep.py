import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.preprocessing import normalize
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

from common import ensure_dir, iter_images, label_from_path, load_config, resolve_project_path


class FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.features = torch.nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)


def load_rgb(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
    return image


def build_model(model_name, device):
    if model_name != "resnet50":
        raise ValueError(f"Unsupported model: {model_name}")
    weights = ResNet50_Weights.DEFAULT
    backbone = resnet50(weights=weights)
    model = FeatureExtractor(backbone).to(device)
    model.eval()
    return model, weights.transforms()


def extract_features(paths, model, preprocess, device, batch_size, title):
    features = []
    valid_paths = []
    failed = []

    batch_images = []
    batch_paths = []

    def flush_batch():
        if not batch_images:
            return
        tensor = torch.stack(batch_images).to(device)
        with torch.inference_mode():
            output = model(tensor).detach().cpu().numpy().astype(np.float32)
        features.append(output)
        valid_paths.extend(batch_paths)
        batch_images.clear()
        batch_paths.clear()

    for path in tqdm(paths, desc=title, unit="img"):
        try:
            image = load_rgb(path)
            batch_images.append(preprocess(image))
            batch_paths.append(path)
            if len(batch_images) >= batch_size:
                flush_batch()
        except Exception as exc:
            failed.append((str(path), str(exc)))

    flush_batch()

    if not features:
        raise RuntimeError(f"No valid images found for {title}.")
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


def save_cache(output_dir, model_name, base_paths, base_features, query_paths, query_features):
    np.savez_compressed(
        output_dir / f"features_{model_name}.npz",
        model_name=model_name,
        base_paths=np.array([str(p) for p in base_paths]),
        base_features=base_features,
        query_paths=np.array([str(p) for p in query_paths]),
        query_features=query_features,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default="outputs/retrieval_resnet50")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-base", type=int, default=None)
    parser.add_argument("--max-query", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    top_k = max(int(k) for k in config["retrieval"]["top_k"])
    output_dir = ensure_dir(resolve_project_path(args.config, args.output_dir))

    device = torch.device(args.device)
    model, preprocess = build_model(args.model, device)

    base_dir = resolve_project_path(args.config, data_cfg["retrieval_base_dir"])
    query_dir = resolve_project_path(args.config, data_cfg["retrieval_query_dir"])
    base_paths = iter_images(base_dir)
    query_paths = iter_images(query_dir)
    if args.max_base:
        base_paths = base_paths[: args.max_base]
    if args.max_query:
        query_paths = query_paths[: args.max_query]

    print("=== Run Deep Feature Retrieval ===")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Base images: {len(base_paths)}")
    print(f"Query images: {len(query_paths)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output dir: {output_dir}")

    base_paths, base_features, base_failed = extract_features(
        base_paths, model, preprocess, device, args.batch_size, f"Encode base {args.model}"
    )
    query_paths, query_features, query_failed = extract_features(
        query_paths, model, preprocess, device, args.batch_size, f"Encode query {args.model}"
    )

    print("Computing similarities...")
    top_indices, top_scores = compute_topk(query_features, base_features, top_k)
    results = build_results_dataframe(query_paths, base_paths, top_indices, top_scores)

    results_path = output_dir / "retrieval_results.csv"
    failures_path = output_dir / "feature_failures.csv"
    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    save_cache(output_dir, args.model, base_paths, base_features, query_paths, query_features)
    failures = pd.DataFrame(base_failed + query_failed, columns=["path", "error"])
    failures.to_csv(failures_path, index=False, encoding="utf-8-sig")

    print(f"Saved results: {results_path}")
    print(f"Saved features: {output_dir / f'features_{args.model}.npz'}")
    print(f"Saved failures: {failures_path} ({len(failures)} rows)")
    print("\nSample Top-5:")
    sample = results[results["query_name"] == results["query_name"].iloc[0]].head(5)
    print(sample[["query_name", "rank", "base_name", "score", "is_relevant"]].to_string(index=False))


if __name__ == "__main__":
    main()
