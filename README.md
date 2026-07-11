# 🎵 Music Recommender System Based on Content-Based Filtering

> Music Recommendation System using Content-Based Filtering with Word Embeddings (GloVe, FastText, Word2Vec)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Overview

This project implements a music recommendation system using a **Content-Based Filtering** approach powered by **Word Embeddings** to represent song lyrics. It compares three distinct word embedding models:

| Model | Hit Rate | MRR | NDCG |
|-------|----------|-----|------|
| 🥇 **GloVe** | **92.65%** | **0.4886** | **0.5792** |
| 🥈 FastText | 90.83% | 0.4692 | 0.5615 |
| 🥉 Word2Vec | 90.44% | 0.4701 | 0.5606 |

**GloVe** is selected as the best-performing model based on a 5-Fold Cross Validation evaluation (K=5) with Top-K=20.

---

## 🗂️ Project Structure

```
research/
│
├── main.ipynb                 ← Main notebook (run this)
│
├── config.py                  ← Path and model parameters configuration
├── preprocessing.py           ← Dataset loading and merging logic
├── models.py                  ← Model downloader and TF-IDF weighted embedding generator
├── evaluation.py              ← K-Fold Cross Validation evaluator
├── visualization.py           ← Performance comparison plotting functions
├── gradio_app.py              ← Interactive UI logic
├── deploy.py                  ← Conversion and automated deployment to Hugging Face
│
├── README.md                  ← Project documentation
├── requirements.txt           ← Python dependencies list
├── .gitignore                 ← Git ignored files configuration
│
├── modelfigures/              ← System architecture diagrams and UI screenshots
└── dataset/
    └── README_dataset.md      ← Dataset information and download links
```

---

## 📊 Dataset & Pre-computed Files

The dataset used in this project is not stored directly in this repository due to file size constraints.
- Detailed download instructions and Kaggle links are available in [`dataset/README_dataset.md`](dataset/README_dataset.md).
- Pre-computed embeddings, trained models, and FAISS index files are shared on Google Drive. You can download them directly to run the project without training:
  👉 **Google Drive Pre-computed Data**: [http://tiny.cc/faiqresearch](http://tiny.cc/faiqresearch)

---

## ⚙️ Methodology

The system pipeline consists of the following phases:

1. **Data Preprocessing** (`preprocessing.py`) – Load, clean, and merge Spotify features dataset with lyrics data.
2. **Word Embedding Representation** (`models.py`) – Represent lyrics text using pre-trained GloVe, FastText, and Word2Vec models.
3. **TF-IDF Weighting** (`models.py`) – Weight embedding vectors based on term importance.
4. **FAISS Indexing** (`models.py`) – Build a fast similarity index using Facebook AI Similarity Search.
5. **K-Fold Evaluation** (`evaluation.py`) – Perform 5-Fold Cross Validation using Hit Rate, MRR, and NDCG metrics.
6. **Visualization** (`visualization.py`) – Generate performance bar charts, radar charts, heatmaps, and per-fold lines.
7. **Gradio UI & Deployment** (`gradio_app.py`, `deploy.py`) – Run a local interactive interface or automatically deploy it to Hugging Face Spaces.

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/faiqmisbah/music-recommendersystem-cbf.git
cd music-recommendersystem-cbf
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Dataset & Models
- Follow the instructions in [`dataset/README_dataset.md`](dataset/README_dataset.md) to download raw datasets, or download pre-computed assets from [http://tiny.cc/faiqresearch](http://tiny.cc/faiqresearch).
- Place them in the corresponding folders: `dataset/lagu/`, `dataset/lirik/`, `dataset/user/`.

### 4. Run the Jupyter Notebook
Open `main.ipynb` in Jupyter Notebook/VS Code and execute the cells sequentially:
```
main.ipynb
├── Step 1: Setup & Import modules
├── Step 2: Load & Preprocessing
├── Step 3: Training & Evaluation (FastText, GloVe, Word2Vec)
├── Step 4: Visualizations
├── Step 5: Summary of Results
├── Step 6: Local Gradio UI Demo
└── Step 7: Deploy to Hugging Face Spaces
```

Alternatively, run individual modules via the terminal:
```bash
python preprocessing.py    # Run dataset load and merge
python models.py           # Run training and evaluation for all models
```

---

## 📦 Requirements

- Python 3.10+
- Jupyter Notebook / VS Code
- Google Drive Desktop (optional, for streaming path configuration)

See `requirements.txt` for the list of Python packages.

---

## 📈 Evaluation Results

Evaluation is carried out using **5-Fold Cross Validation** with **Top-K=20** and **Related Genre Matching**:

| Metric | GloVe | FastText | Word2Vec |
|--------|-------|----------|----------|
| **Hit Rate** | **92.65%** | 90.83% | 90.44% |
| **MRR** | **0.4886** | 0.4692 | 0.4701 |
| **NDCG** | **0.5792** | 0.5615 | 0.5606 |

---

## 📁 Module Reference

| File | Description |
|------|-------------|
| `config.py` | Global configuration of directory paths, model hyperparameters, and genre groups |
| `preprocessing.py` | String cleaning, normalizations, and data-merging routines for CSV sources |
| `models.py` | Pre-trained model loading, TF-IDF embedding computation, and FAISS indexing |
| `evaluation.py` | Implementation of K-Fold CV, Hit Rate, MRR, and NDCG metrics |
| `visualization.py` | Charting functions (bar, radar, heatmap, fold comparisons) |
| `gradio_app.py` | Local and cloud-compatible Gradio Blocks UI layout |
| `deploy.py` | Automatic packaging and deployment wrapper for Hugging Face Spaces |
| `main.ipynb` | Main entry point notebook orchestrating all modules |

---

## 👤 Author

**Faiq Misbah Yazdi**
Undergraduate Thesis / Final Project — 2025

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
