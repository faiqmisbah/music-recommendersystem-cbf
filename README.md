# 🎵 Music Recommender System Based on Content-Based Filtering with GloVe

> Sistem Rekomendasi Musik Berbasis Content-Based Filtering menggunakan Word Embedding (GloVe, FastText, Word2Vec)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Overview / Gambaran Umum

Proyek ini merupakan implementasi sistem rekomendasi musik berbasis **Content-Based Filtering** yang memanfaatkan teknik **Word Embedding** untuk merepresentasikan lirik lagu. Penelitian ini membandingkan tiga model word embedding:

| Model | Hit Rate | MRR | NDCG |
|-------|----------|-----|------|
| 🥇 **GloVe** | **92.65%** | **0.4886** | **0.5792** |
| 🥈 FastText | 90.83% | 0.4692 | 0.5615 |
| 🥉 Word2Vec | 90.44% | 0.4701 | 0.5606 |

**GloVe** terpilih sebagai model terbaik berdasarkan evaluasi K-Fold Cross Validation (K=5) dengan Top-K=20.

---

## 🗂️ Project Structure / Struktur Proyek

```
research/
│
├── main.ipynb                 ← Notebook utama (jalankan ini)
│
├── config.py                  ← Konfigurasi path & parameter
├── preprocessing.py           ← Load & merge dataset
├── models.py                  ← Load model, generate embeddings
├── evaluation.py              ← K-Fold Cross Validation
├── visualization.py           ← Visualisasi perbandingan model
│
├── README.md                  ← Dokumentasi proyek ini
├── requirements.txt           ← Daftar dependensi Python
├── .gitignore                 ← File yang diabaikan Git
│
└── dataset/
    └── README_dataset.md      ← Info & tautan sumber dataset
```

---

## 📊 Dataset

Dataset yang digunakan dalam penelitian ini:

1. **Spotify Features** – Audio features dan metadata lagu dari Spotify
2. **Song Lyrics** – Lirik lagu dari berbagai artis
3. **User Top Tracks** – Data riwayat lagu yang disukai pengguna

> Dataset tidak disertakan di repositori ini. Lihat detail di [`dataset/README_dataset.md`](dataset/README_dataset.md)

---

## ⚙️ Metodologi / Methodology

Alur penelitian terdiri dari beberapa tahap:

1. **Data Preprocessing** (`preprocessing.py`) – Load, normalisasi, dan merge dataset lagu + lirik
2. **Word Embedding** (`models.py`) – Representasi lirik menggunakan GloVe, FastText, Word2Vec
3. **TF-IDF Weighting** (`models.py`) – Pembobotan kata menggunakan TF-IDF
4. **FAISS Indexing** (`models.py`) – Pencarian similaritas menggunakan Facebook AI Similarity Search
5. **K-Fold Evaluation** (`evaluation.py`) – Evaluasi 5-Fold CV: Hit Rate, MRR, NDCG
6. **Visualisasi** (`visualization.py`) – Bar chart, radar chart, heatmap perbandingan
7. **UI Demo** (`main.ipynb`) – Antarmuka Gradio untuk demonstrasi

---

## 🚀 How to Run / Cara Menjalankan

### 1. Clone Repositori
```bash
git clone https://github.com/[username]/music-recommender-glove.git
cd music-recommender-glove
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Dataset
- Download dataset sesuai instruksi di [`dataset/README_dataset.md`](dataset/README_dataset.md)
- Letakkan di folder `dataset/lagu/`, `dataset/lirik/`, `dataset/user/`

### 4. Jalankan Notebook
Buka `main.ipynb` di Jupyter Notebook atau VS Code, lalu jalankan cell dari atas ke bawah.

```
main.ipynb
├── Tahap 1: Setup & Import modul
├── Tahap 2: Load & Preprocessing
├── Tahap 3: Training & Evaluasi (FastText, GloVe, Word2Vec)
├── Tahap 4: Visualisasi Perbandingan
├── Tahap 5: Ringkasan Hasil
└── Tahap 6: UI Gradio Demo
```

Atau jalankan modul secara terpisah dari terminal:
```bash
python preprocessing.py    # Load & merge dataset
python models.py           # Training & evaluasi semua model
```

---

## 📦 Requirements

- Python 3.10+
- Jupyter Notebook / VS Code
- Google Drive Desktop (untuk akses dataset)

Lihat `requirements.txt` untuk daftar lengkap library.

---

## 📈 Hasil Evaluasi / Evaluation Results

Evaluasi menggunakan **5-Fold Cross Validation** dengan **Top-K=20** dan **Related Genre Matching**:

| Metrik | GloVe | FastText | Word2Vec |
|--------|-------|----------|----------|
| Hit Rate | 92.65% | 90.83% | 90.44% |
| MRR | 0.4886 | 0.4692 | 0.4701 |
| NDCG | 0.5792 | 0.5615 | 0.5606 |

---

## 📁 Penjelasan Modul / Module Description

| File | Deskripsi |
|------|-----------|
| `config.py` | Konfigurasi path, parameter model (Top-K, K-Folds), genre groups |
| `preprocessing.py` | Fungsi cleaning metadata & lirik, load & merge 3 dataset CSV |
| `models.py` | Load pre-trained model (gensim), TF-IDF weighted embeddings, FAISS index |
| `evaluation.py` | K-Fold CV dengan metrik Hit Rate, MRR, NDCG |
| `visualization.py` | Bar chart, radar chart, heatmap, per-fold line chart |
| `main.ipynb` | Notebook utama yang mengimpor semua modul di atas |

---

## 👤 Author / Penulis

**Faiq Misbah Yazdi**
Tugas Akhir / Skripsi — 2025

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
