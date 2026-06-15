# Computer Vision Lab 2 Project

This project is the working folder for `实验2-260517.pptx`.

## Goal

Finish two required tasks:

1. Image retrieval: use query images to retrieve similar images from the base image database.
2. Text/object detection visualization: show text detection boxes for retrieved BJTU landmark images.

## Data

The project does not copy the raw dataset. The default config points to:

- `../image_retrieval/base`
- `../image_retrieval/query`
- `../object_detection/data`

Run commands from this folder:

```powershell
cd D:\计算机视觉基础\cv_lab2_project
```

## Environment

The recommended local virtual environment is `.venv`.

Activate it before running scripts:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies if the environment needs to be rebuilt:

```powershell
python -m pip install -r requirements.txt
```

Installed acceleration libraries:

- `numpy`: vectorized feature extraction and similarity calculation.
- `pandas`: result tables and CSV files.
- `pillow`: image loading, resizing, drawing and contact sheets.
- `opencv-python`: image processing, ORB/SIFT-style feature experiments.
- `scikit-learn`: nearest-neighbor retrieval, normalization and metrics.
- `matplotlib`: Precision@K figures for the report.
- `tqdm`: progress bars for long image loops.
- `pyyaml`: config loading.

## Step-By-Step Workflow

### Step 0. Check Data

Confirm that images and annotations can be read.

```powershell
python src/check_data.py --config configs/config.yaml
```

Expected output:

- base image count
- query image count
- object detection image/json count
- landmark distribution
- warning list for missing image/json pairs

### Step 1. Build Retrieval Baseline

Run the implemented accelerated baseline:

```powershell
python src/run_retrieval.py --config configs/config.yaml
```

It extracts HSV color histogram + grayscale thumbnail structure features with OpenCV/NumPy, then uses cosine similarity to retrieve Top-60 results.

Debug run with a small subset:

```powershell
python src/run_retrieval.py --config configs/config.yaml --max-base 300 --max-query 5
```

Expected outputs:

- `outputs/retrieval/features_color_struct.npz`
- `outputs/retrieval/retrieval_results.csv`
- `outputs/retrieval/feature_failures.csv`

### Step 2. Evaluate Precision@K

Compute P@20, P@40 and P@60 by filename prefix.

```powershell
python src/evaluate_retrieval.py --config configs/config.yaml
```

Expected outputs:

- `outputs/retrieval/precision_at_k.csv`
- `outputs/retrieval/precision_at_k_by_query.csv`
- `outputs/figures/p_at_k_<landmark>.png`
- `outputs/figures/p_at_k_all_landmarks.png`

Current color+structure baseline summary:

- Overall P@20: 0.1381
- Overall P@40: 0.1019
- Overall P@60: 0.0896

This is a runnable baseline, but the retrieval quality is weak. A stronger feature method should be added before final submission if time allows.

### Step 2b. Build SIFT-BoVW Retrieval

Run the SIFT Bag-of-Visual-Words method:

```powershell
python src/run_retrieval_sift_bovw.py --config configs/config.yaml --output-dir outputs/retrieval_sift --vocab-size 256 --dictionary-max-images 2000 --dictionary-max-descriptors 120000 --image-size 640 --max-keypoints 500 --max-descriptors-per-image 120
```

Evaluate SIFT-BoVW:

```powershell
python src/evaluate_retrieval.py --config configs/config.yaml --results outputs/retrieval_sift/retrieval_results.csv --output-dir outputs/retrieval_sift --figures-dir outputs/figures_sift
```

Expected outputs:

- `outputs/retrieval_sift/retrieval_results.csv`
- `outputs/retrieval_sift/features_sift_bovw.npz`
- `outputs/retrieval_sift/precision_at_k.csv`
- `outputs/figures_sift/p_at_k_<landmark>.png`
- `outputs/figures_sift/p_at_k_all_landmarks.png`

SIFT-BoVW summary:

- Overall P@20: 0.1856
- Overall P@40: 0.1526
- Overall P@60: 0.1400

Compared with the color+structure baseline, SIFT-BoVW improves all overall P@K scores and is the better retrieval method for the report.

### Step 2c. Build Deep Feature Retrieval

Run ImageNet-pretrained ResNet50 feature retrieval:

```powershell
python src/run_retrieval_deep.py --config configs/config.yaml --output-dir outputs/retrieval_resnet50 --model resnet50 --batch-size 16
```

Evaluate ResNet50:

```powershell
python src/evaluate_retrieval.py --config configs/config.yaml --results outputs/retrieval_resnet50/retrieval_results.csv --output-dir outputs/retrieval_resnet50 --figures-dir outputs/figures_resnet50
```

Expected outputs:

- `outputs/retrieval_resnet50/retrieval_results.csv`
- `outputs/retrieval_resnet50/features_resnet50.npz`
- `outputs/retrieval_resnet50/precision_at_k.csv`
- `outputs/figures_resnet50/p_at_k_<landmark>.png`

ResNet50 uses ImageNet-pretrained features only. It does not use this dataset's landmark labels for training, so labels are still used only for evaluation.

### Step 2d. Compare Retrieval Methods

Generate a comparison table and figures for all three retrieval methods:

```powershell
python src/compare_methods.py --config configs/config.yaml --output-dir outputs/comparison
```

Current overall comparison:

| Method | P@20 | P@40 | P@60 |
| --- | ---: | ---: | ---: |
| Color+Structure | 0.1381 | 0.1019 | 0.0896 |
| SIFT-BoVW | 0.1856 | 0.1526 | 0.1400 |
| ResNet50 | 0.8311 | 0.7874 | 0.7531 |

The ResNet50 feature extractor is the strongest retrieval method and should be used as the main result in the report.

### Step 3. Visualize Retrieval Results

Create contact sheets for selected query images.

```powershell
python src/visualize_retrieval.py --config configs/config.yaml --results outputs/retrieval_sift/retrieval_results.csv --output-dir outputs/demo_cases_sift --top-n 5 --cases-per-landmark 2
```

Expected outputs:

- `outputs/demo_cases_sift/<landmark>_case_<n>_<query_stem>_retrieval.jpg`
- `outputs/demo_cases_sift/manifest.csv`

Visualization legend:

- Blue border: query image.
- Green border: retrieved image with the same filename prefix.
- Red border: retrieved image with a different filename prefix.

### Step 4. Visualize Text Detection

Use LabelMe json boxes as the reliable text detection visualization baseline.

```powershell
python src/visualize_detection.py --config configs/config.yaml --manifest outputs/demo_cases_resnet50/manifest.csv --output-dir outputs/detection_resnet50 --combined-dir outputs/demo_retrieval_detection_resnet50
```

Expected outputs:

- `outputs/detection_resnet50/<landmark>_case_<n>_<query_stem>_detection.jpg`
- `outputs/demo_retrieval_detection_resnet50/<landmark>_case_<n>_<query_stem>_retrieval_detection.jpg`
- `outputs/demo_retrieval_detection_resnet50/manifest.csv`

Current output contains 24 aligned retrieval-detection cases: 12 landmarks x 2 cases.

### Step 5. Prepare Report Assets

Collect result tables and 24 visualization groups for the report.

```powershell
python src/prepare_report_assets.py --config configs/config.yaml
```

Expected outputs:

- `outputs/report_assets/tables`
- `outputs/report_assets/figures`
- `outputs/report_assets/cases_retrieval_detection`
- `outputs/report_assets/cases_retrieval_only`
- `outputs/report_assets/cases_detection_only`
- `outputs/report_assets/asset_manifest.csv`
- `outputs/report_assets/asset_summary.csv`

Current report assets contain 95 items, including:

- 5 result tables
- 16 figures
- 24 retrieval-detection cases
- 24 retrieval-only cases
- 24 detection-only cases

## Submission Checklist

- PDF experiment report
- Source code repository link
- README with run commands
- Demo video with 3-5 test examples
- No source code pasted into the report

## Demo Video

The experiment demo video is stored at:

- `docs/demo_video.mp4`
