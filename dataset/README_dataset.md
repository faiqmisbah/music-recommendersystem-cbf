# 📊 Dataset Information / Informasi Dataset

Dataset yang digunakan dalam penelitian ini **tidak disertakan** di repositori ini karena keterbatasan ukuran file dan lisensi. Silakan download dari sumber berikut dan letakkan di folder yang sesuai.

---

## 1. 🎵 Spotify Audio Features (`SpotifyFeatures.csv`)

**Sumber / Source:**
> Kaggle — Spotify Dataset 1921-2020, 600k+ Tracks
> 🔗 https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks

**Deskripsi:** Dataset berisi audio features dari jutaan lagu di Spotify, termasuk:
- `track_name` — Judul lagu
- `artist_name` — Nama artis
- `genre` — Genre musik
- `popularity` — Tingkat popularitas (0–100)
- `danceability`, `energy`, `valence`, dll. — Fitur audio

**Letakkan di:** `dataset/lagu/SpotifyFeatures.csv`

---

## 2. 📝 Song Lyrics (`lyrics-data.csv`)

**Sumber / Source:**
> Kaggle — Song Lyrics Dataset
> 🔗 https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres

**Deskripsi:** Dataset berisi lirik lagu dari berbagai artis dan genre:
- `SName` — Judul lagu
- `ALink` — Link artis
- `Lyric` — Teks lirik lagu

**Letakkan di:** `dataset/lirik/lyrics-data.csv`

---

## 3. 👤 User Top Tracks (`user_top_tracks.csv`)

**Sumber / Source:**
> Data dikumpulkan dari Spotify User History (user study)

**Deskripsi:** Dataset riwayat lagu yang disukai pengguna:
- `user_id` — ID pengguna
- `liked_song_indices` — Indeks lagu yang disukai

**Letakkan di:** `dataset/user/user_top_tracks.csv`

---

## 📁 Struktur Folder Dataset

Setelah download, susun folder seperti berikut:

```
research/
└── dataset/
    ├── lagu/
    │   └── SpotifyFeatures.csv       ← Taruh di sini
    ├── lirik/
    │   └── lyrics-data.csv           ← Taruh di sini
    └── user/
        └── user_top_tracks.csv       ← Taruh di sini
```

---

## 📊 Statistik Dataset

| Dataset | Jumlah Record | Ukuran File |
|---------|--------------|-------------|
| SpotifyFeatures.csv | ~232,000 lagu | ~43 MB |
| lyrics-data.csv | ~379,000 lirik | ~330 MB |
| user_top_tracks.csv | ~100 user | <1 MB |
| **Setelah merge** | **~20,101 lagu** | — |

---

> ⚠️ **Catatan:** Dataset `lyrics-data.csv` berukuran cukup besar. Pastikan koneksi internet stabil saat mendownload dari Kaggle.
