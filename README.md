# 计算机视觉基础实验二：校园地标图像检索与检测可视化

本项目用于完成《计算机视觉基础》实验二，主要包含两个部分：

1. 图像检索：给定 query 图像，在 base 图像库中检索相似校园地标图像，并使用 P@20、P@40、P@60 进行评价。
2. 检测可视化：读取数据集中提供的标注 JSON，在图像上绘制检测框，展示检索结果与检测结果。

最终主要方案采用 ImageNet 预训练 ResNet50 提取深度特征进行图像检索。该方法没有使用本实验数据集的类别标签进行训练，类别标签仅由文件名前缀解析，并只用于最终评价。

## 项目结构

```text
cv_lab2_project/
├── configs/
│   └── config.yaml
├── docs/
│   ├── demo_video.mp4
│   ├── requirements_breakdown.md
│   └── report_assets_preview/
├── src/
│   ├── check_data.py
│   ├── run_retrieval.py
│   ├── run_retrieval_sift_bovw.py
│   ├── run_retrieval_deep.py
│   ├── evaluate_retrieval.py
│   ├── compare_methods.py
│   ├── visualize_retrieval.py
│   ├── visualize_detection.py
│   └── prepare_report_assets.py
├── requirements.txt
└── README.md
```

原始数据集和虚拟环境未上传到 GitHub，避免仓库体积过大。默认配置假设数据集位于项目上一级目录：

```text
../image_retrieval/base
../image_retrieval/query
../object_detection/data
```

## 环境配置

进入项目目录：

```powershell
cd D:\计算机视觉基础\cv_lab2_project
```

创建并激活虚拟环境后安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

主要使用的库包括：

| 库 | 用途 |
| --- | --- |
| NumPy | 特征向量计算、矩阵相似度计算 |
| Pandas | 检索结果和评价指标表格处理 |
| Pillow | 图像读取、绘制、拼接可视化结果 |
| OpenCV | 传统特征、SIFT/BoVW 特征实验 |
| scikit-learn | 特征归一化、KMeans、评价辅助 |
| Matplotlib | 绘制 P@K 曲线和方法对比图 |
| PyTorch / Torchvision | 加载 ResNet50 并提取深度特征 |
| tqdm | 长任务进度显示 |
| PyYAML | 读取配置文件 |

## 实验方法

本项目实现并比较了三种图像检索方法：

| 方法 | 简介 |
| --- | --- |
| Color+Structure | HSV 颜色直方图 + 灰度缩略图结构特征 |
| SIFT-BoVW | SIFT 局部特征 + Bag of Visual Words 词袋表示 |
| ResNet50 | ImageNet 预训练 ResNet50 深度特征 |

其中 ResNet50 是最终采用的主方法，原因是它对建筑地标的语义、纹理和局部结构表达更强，检索准确率明显高于传统手工特征。

## 运行步骤

检查数据集路径和标注文件：

```powershell
python src\check_data.py --config configs\config.yaml
```

运行颜色和结构特征基线：

```powershell
python src\run_retrieval.py --config configs\config.yaml
python src\evaluate_retrieval.py --config configs\config.yaml
```

运行 SIFT-BoVW 检索：

```powershell
python src\run_retrieval_sift_bovw.py --config configs\config.yaml --output-dir outputs\retrieval_sift --vocab-size 256 --dictionary-max-images 2000 --dictionary-max-descriptors 120000 --image-size 640 --max-keypoints 500 --max-descriptors-per-image 120
python src\evaluate_retrieval.py --config configs\config.yaml --results outputs\retrieval_sift\retrieval_results.csv --output-dir outputs\retrieval_sift --figures-dir outputs\figures_sift
```

运行 ResNet50 深度特征检索：

```powershell
python src\run_retrieval_deep.py --config configs\config.yaml --output-dir outputs\retrieval_resnet50 --model resnet50 --batch-size 16
python src\evaluate_retrieval.py --config configs\config.yaml --results outputs\retrieval_resnet50\retrieval_results.csv --output-dir outputs\retrieval_resnet50 --figures-dir outputs\figures_resnet50
```

生成三种方法对比结果：

```powershell
python src\compare_methods.py --config configs\config.yaml --output-dir outputs\comparison
```

生成检索样例图和检测可视化图：

```powershell
python src\visualize_retrieval.py --config configs\config.yaml --results outputs\retrieval_resnet50\retrieval_results.csv --output-dir outputs\demo_cases_resnet50 --top-n 5 --cases-per-landmark 2
python src\visualize_detection.py --config configs\config.yaml --manifest outputs\demo_cases_resnet50\manifest.csv --output-dir outputs\detection_resnet50 --combined-dir outputs\demo_retrieval_detection_resnet50
```

整理报告素材：

```powershell
python src\prepare_report_assets.py --config configs\config.yaml
```

## 实验结果

三种检索方法的整体评价结果如下：

| 方法 | P@20 | P@40 | P@60 |
| --- | ---: | ---: | ---: |
| Color+Structure | 0.1381 | 0.1019 | 0.0896 |
| SIFT-BoVW | 0.1856 | 0.1526 | 0.1400 |
| ResNet50 | 0.8311 | 0.7874 | 0.7531 |

可以看到，ResNet50 在 P@20、P@40、P@60 三个指标上均取得最高结果，最终作为本实验的主要检索方案。

部分报告预览素材保存在：

```text
docs/report_assets_preview/
```

其中包含方法对比图和若干检索-检测联合可视化样例。

## 检测可视化

检测部分使用数据集中提供的 LabelMe JSON 标注文件。程序读取标注中的目标框，并绘制到对应图像上，用于展示文本或目标区域的检测结果。

检测可视化脚本会生成两类结果：

1. 单独的检测框图像。
2. 检索结果与检测框结合的联合展示图像。

最终生成了 24 组检索-检测联合样例，对应 12 类地标，每类 2 组。

## 演示视频

演示视频已放在仓库中：

```text
docs/demo_video.mp4
```

视频内容包括项目结构、实验任务、三种检索方法对比、ResNet50 检索效果、P@K 指标结果，以及检索和检测可视化样例。

## 说明

- 本项目不上传原始数据集。
- 本项目不上传 `.venv` 虚拟环境。
- `outputs/` 目录中的大规模中间结果和缓存文件默认不纳入 Git。
- 类别标签仅用于评价，不参与模型训练。
