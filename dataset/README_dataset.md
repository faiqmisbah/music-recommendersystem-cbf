# 📊 Dataset Information

The raw dataset used in this research is **not stored** directly in this repository due to file size constraints and licensing terms. You can download the source datasets from the Kaggle links below, or download the pre-processed assets directly from Google Drive.

---

## 🚀 Pre-processed Data & Model Assets (Recommended)

If you wish to run the notebook/Gradio UI directly without downloading the massive raw CSVs and retraining the word embeddings (which takes time and resources), you can download the pre-processed datasets, trained embeddings, and FAISS index files from:
👉 **Google Drive Repository**: [http://tiny.cc/faiqresearch](http://tiny.cc/faiqresearch)

Download the files from the link above and place them in the following directories:
* Files from `data/` folder (`merged_songs.csv`, `user_profiles.csv`) → Place under `research/dataset/` or `huggingface_deploy/data/`
* Files from `embeddings/` folder (`*_embeddings_tfidf.npy`) → Place under `research/embeddings/` or `huggingface_deploy/embeddings/`

---

## 📥 Raw Datasets from Source (Kaggle)

If you prefer to preprocess the data from scratch, download these raw datasets:

### 1. 🎵 Spotify Tracks (`SpotifyFeatures.csv`)
* **Source:** Kaggle — Ultimate Spotify Tracks DB
* **Link:** [https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db)
* **Description:** Contains audio features and metadata for ~232k Spotify songs.
* **Save as:** `research/dataset/lagu/SpotifyFeatures.csv`

### 2. 📝 Song Lyrics (`lyrics-data.csv`)
* **Source:** Kaggle — Song Lyrics Dataset (Scrapped lyrics from 6 genres)
* **Link:** [https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres](https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres)
* **Description:** Contains song lyrics across multiple genres.
* **Save as:** `research/dataset/lirik/lyrics-data.csv`

### 3. 👤 User Music History (`user_top_tracks.csv`)
* **Source:** Kaggle — Music Listening Data (500k Users)
* **Link:** [https://www.kaggle.com/datasets/gabrielkahen/music-listening-data-500k-users](https://www.kaggle.com/datasets/gabrielkahen/music-listening-data-500k-users)
* **Description:** Contains user listening history and profiles.
* **Save as:** `research/dataset/user/user_top_tracks.csv`

---

## 📁 Dataset Folder Structure

After downloading, arrange your directories as follows:

```
research/
└── dataset/
    ├── lagu/
    │   └── SpotifyFeatures.csv
    ├── lirik/
    │   └── lyrics-data.csv
    └── user/
        └── user_top_tracks.csv
```

---

## 📊 Dataset Statistics

| Dataset | Records / Rows | File Size |
|---------|----------------|-----------|
| SpotifyFeatures.csv | ~232,000 | ~43 MB |
| lyrics-data.csv | ~379,000 | ~330 MB |
| user_top_tracks.csv | ~100 users | <1 MB |
| **Merged Dataset (Valid)** | **~20,101 songs** | — |
