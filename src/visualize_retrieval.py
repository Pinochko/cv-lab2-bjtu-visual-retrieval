import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

from common import ensure_dir, load_config, resolve_project_path


def load_font(size=18):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def open_square_thumbnail(path, size):
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def draw_tile(path, size, title, subtitle, border_color, font, small_font):
    pad = 8
    text_h = 52
    tile = Image.new("RGB", (size + pad * 2, size + text_h + pad * 3), "white")
    thumb = open_square_thumbnail(path, size)
    thumb = ImageOps.expand(thumb, border=4, fill=border_color)
    tile.paste(thumb, (pad, pad))

    draw = ImageDraw.Draw(tile)
    draw.text((pad, size + pad * 2), title, fill=(20, 20, 20), font=font)
    draw.text((pad, size + pad * 2 + 24), subtitle, fill=(80, 80, 80), font=small_font)
    return tile


def make_contact_sheet(query_row, retrieved_rows, output_path, top_n, tile_size):
    font = load_font(17)
    small_font = load_font(13)
    gap = 10
    bg = (245, 247, 250)
    good = (37, 135, 75)
    bad = (203, 73, 73)
    query_color = (45, 96, 180)

    tiles = []
    query_path = Path(query_row["query_path"])
    query_tile = draw_tile(
        query_path,
        tile_size,
        "Query",
        query_row["query_name"],
        query_color,
        font,
        small_font,
    )
    tiles.append(query_tile)

    for _, row in retrieved_rows.head(top_n).iterrows():
        border = good if int(row["is_relevant"]) == 1 else bad
        title = f"Top {int(row['rank'])} | score {float(row['score']):.3f}"
        subtitle = row["base_name"]
        tiles.append(draw_tile(Path(row["base_path"]), tile_size, title, subtitle, border, font, small_font))

    width = sum(tile.width for tile in tiles) + gap * (len(tiles) + 1)
    height = max(tile.height for tile in tiles) + gap * 2
    sheet = Image.new("RGB", (width, height), bg)

    x = gap
    for tile in tiles:
        sheet.paste(tile, (x, gap))
        x += tile.width + gap

    ensure_dir(output_path.parent)
    sheet.save(output_path, quality=92)


def select_cases(results, landmark_prefixes, cases_per_landmark):
    selected = []
    for label in landmark_prefixes:
        query_names = (
            results[results["query_label"] == label]
            .groupby("query_name")["is_relevant"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )
        for query_name in query_names[:cases_per_landmark]:
            selected.append((label, query_name))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--results", default="outputs/retrieval_sift/retrieval_results.csv")
    parser.add_argument("--output-dir", default="outputs/demo_cases_sift")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--cases-per-landmark", type=int, default=2)
    parser.add_argument("--tile-size", type=int, default=190)
    args = parser.parse_args()

    config = load_config(args.config)
    landmark_prefixes = [str(v).lower() for v in config["data"]["landmark_prefixes"]]
    results_path = resolve_project_path(args.config, args.results)
    output_dir = ensure_dir(resolve_project_path(args.config, args.output_dir))

    if not results_path.exists():
        raise FileNotFoundError(f"Missing retrieval results: {results_path}")

    results = pd.read_csv(results_path)
    cases = select_cases(results, landmark_prefixes, args.cases_per_landmark)

    manifest_rows = []
    print("=== Visualize Retrieval Results ===")
    print(f"Results: {results_path}")
    print(f"Output dir: {output_dir}")
    print(f"Cases: {len(cases)}")

    for label, query_name in tqdm(cases, desc="Create contact sheets", unit="case"):
        query_rows = results[results["query_name"] == query_name].sort_values("rank")
        if query_rows.empty:
            continue
        query_row = query_rows.iloc[0]
        case_index = sum(1 for row in manifest_rows if row["label"] == label) + 1
        output_path = output_dir / f"{label}_case_{case_index}_{Path(query_name).stem}_retrieval.jpg"
        make_contact_sheet(query_row, query_rows, output_path, args.top_n, args.tile_size)
        precision_top_n = float(query_rows.head(args.top_n)["is_relevant"].mean())
        manifest_rows.append(
            {
                "label": label,
                "case_index": case_index,
                "query_name": query_name,
                "output_path": str(output_path),
                f"P@{args.top_n}": precision_top_n,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved {len(manifest)} contact sheets.")


if __name__ == "__main__":
    main()
