# Multi-Modal Skin Lesion Classifier (Swin Transformer + Clinical Metadata Fusion)

A deep learning system for classifying skin lesions from dermatoscopic images, trained on the **HAM10000** dataset. Unlike standard image-only classifiers, this model fuses **visual features** (via a Swin Transformer) with **clinical metadata** (patient age, sex, lesion localization) to produce more clinically-grounded predictions, and includes **LIME-based explainability** to visualize what the model is "looking at."

## 🧠 Overview

- **Task**: 7-class skin lesion classification (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`)
- **Architecture**: Dual-branch fusion network
  - **Vision branch**: `microsoft/swin-tiny-patch4-window7-224` (Swin Transformer), pretrained, used as a feature extractor
  - **Metadata branch**: A small MLP over encoded patient age, sex, and lesion location
  - **Fusion**: Concatenated visual (768-d) + metadata (32-d) features → classification head
- **Class imbalance handling**: Custom **Focal Loss** implementation
- **Explainability**: **LIME** (Local Interpretable Model-agnostic Explanations) heatmaps over predictions
- **Dataset**: [HAM10000 ("Human Against Machine")](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) via Kaggle

## 🏗️ Architecture

```
Image ──► Swin Transformer ──► 768-d features ─┐
                                                 ├─► Concat (800-d) ─► Classifier ─► 7 classes
Metadata ──► MLP (Age, Sex, Location) ─► 32-d ──┘
```

## 📁 Repository Structure

```
skin-lesion-classifier/
├── notebooks/
│   └── skin_lesion_classifier.ipynb   # Full end-to-end pipeline (Colab-based)
├── results/                           # Generated figures/tables (populated after running)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## ⚙️ Pipeline (as implemented in the notebook)

1. **Environment Setup & Data Download** — Installs dependencies, pulls HAM10000 via the Kaggle API
2. **Data Preprocessing** — Maps image IDs to file paths, imputes missing ages, one-hot encodes sex/localization, standard-scales age
3. **Data Augmentation & Dataset Class** — Custom PyTorch `Dataset` with horizontal/vertical flips + color jitter for training
4. **Model Architecture** — `MultiModalSkinCancerModel` (Swin + MLP fusion, defined above)
5. **Focal Loss** — Custom loss function to handle class imbalance across the 7 diagnosis categories
6. **Training Loop** — AdamW optimizer, 10 epochs, GPU-accelerated
7. **Evaluation** — Classification report + confusion matrix on the held-out validation split
8. **Sanity Check** — Random single-patient inference test
9. **Research Artifacts** — Auto-generates publication-ready figures/tables (learning curves, confusion matrix, classification report image)
10. **Explainable AI (LIME)** — Visual heatmap explaining which image regions drove a given prediction

## 🚀 Getting Started

### Option 1: Run in Google Colab (recommended, no local GPU needed)
1. Open `notebooks/skin_lesion_classifier.ipynb` in Google Colab
2. Run cells top to bottom
3. When prompted, upload your **Kaggle API token** (`kaggle.json`) — get one from [kaggle.com/settings](https://www.kaggle.com/settings) → "Create New Token"

### Option 2: Run locally
```bash
git clone https://github.com/<your-username>/skin-lesion-classifier.git
cd skin-lesion-classifier
pip install -r requirements.txt
jupyter notebook notebooks/skin_lesion_classifier.ipynb
```
You'll need a Kaggle API token placed at `~/.kaggle/kaggle.json` to download the dataset, and a CUDA-capable GPU is strongly recommended for training.

## 📊 Results

After running the evaluation cells, the notebook generates:
- `Table_1_Classification_Report.csv` / `.png` — per-class precision, recall, F1-score
- `Figure_2_Confusion_Matrix.png` — confusion matrix heatmap
- `LIME_Explanation.png` — visual explainability overlay
- `Focal_Loss_Formula.png` — formula reference image

*(Move generated artifacts into `results/` to keep them organized, or update the notebook's save paths.)*

## 🔧 Tech Stack

- PyTorch & Torchvision
- Hugging Face `transformers` (Swin Transformer)
- scikit-learn (preprocessing, metrics, train/test split)
- LIME & SHAP (explainability)
- pandas, NumPy, Matplotlib, Seaborn
- Kaggle API (dataset download)

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- HAM10000 dataset creators (Tschandl, Rosendahl, Kittler)
- Microsoft Research (Swin Transformer)
- Hugging Face `transformers` team
