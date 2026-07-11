"""
config.py — Konfigurasi global untuk sistem rekomendasi musik.
"""

import os

# =====================================================
# PATH CONFIGURATION
# Sesuaikan BASE_PATH dengan lokasi Google Drive Anda
# =====================================================
BASE_PATH = r"G:\My Drive\FinalArc\research"

# Sub-paths
DATASET_PATH = os.path.join(BASE_PATH, "dataset")
MODELS_PATH = os.path.join(BASE_PATH, "saved_models")
EMBEDDINGS_PATH = os.path.join(BASE_PATH, "embeddings")
VISUALISASI_PATH = os.path.join(BASE_PATH, "visualisasi")

# Buat folder jika belum ada
for path in [MODELS_PATH, EMBEDDINGS_PATH, VISUALISASI_PATH]:
    os.makedirs(path, exist_ok=True)

# =====================================================
# MODEL PARAMETERS
# =====================================================
EMBEDDING_DIM = 300
TOP_K = 20
K_FOLDS = 5
RANDOM_SEED = 42

# Genre groups untuk related genre matching (evaluasi)
GENRE_GROUPS = {
    'Rock': ['Rock', 'Alternative', 'Indie', 'Punk', 'Metal', 'Grunge'],
    'Pop': ['Pop', 'Dance', 'Electronic', 'Synth-pop'],
    'Hip-Hop': ['Hip-Hop', 'Rap', 'R&B', 'Soul'],
    'Country': ['Country', 'Folk', 'Americana', 'Bluegrass'],
    'Jazz': ['Jazz', 'Blues', 'Soul'],
    'Electronic': ['Electronic', 'Dance', 'House', 'Techno', 'EDM'],
    'Classical': ['Classical', 'Opera', 'Orchestral'],
    'Latin': ['Latin', 'Reggaeton', 'Salsa'],
    'Reggae': ['Reggae', 'Ska', 'Dub'],
}
