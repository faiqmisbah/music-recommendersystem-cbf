"""
preprocessing.py — Fungsi-fungsi untuk load dan preprocessing dataset.

Modul ini berisi:
- clean_text_metadata(): Cleaning judul lagu & nama artis
- extract_artist_from_alink(): Ekstrak artis dari URL
- clean_lyrics_content(): Cleaning lirik untuk embedding
- clean_lyrics_universal(): Cleaning universal untuk semua model
- load_and_merge_dataset(): Load ketiga CSV & merge menjadi satu DataFrame
"""

import os
import re
import pandas as pd
from config import BASE_PATH, DATASET_PATH


# =====================================================
# FUNGSI CLEANING
# =====================================================

def clean_text_metadata(text):
    """Membersihkan Judul Lagu & Nama Artis untuk keperluan Matching/Merging."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_artist_from_alink(link):
    """Mengambil nama artis dari link URL (khusus dataset lirik)."""
    if pd.isna(link):
        return ""
    name = link.strip("/").split("/")[-1]
    name = re.sub(r'[-_]', ' ', name)
    return name.lower().strip()


def clean_lyrics_content(text):
    """Membersihkan Lirik untuk keperluan Embedding (mempertahankan struktur kata)."""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_lyrics_universal(text):
    """
    Membersihkan teks lirik (versi universal untuk semua model).
    Digunakan saat generate embeddings.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =====================================================
# FUNGSI LOAD & MERGE DATASET
# =====================================================

def load_and_merge_dataset(save_merged=True):
    """
    Load ketiga dataset (Spotify, Lirik, User) dan merge menjadi satu DataFrame.
    
    Returns:
        merged_df (DataFrame): Dataset gabungan lagu + lirik
        user_df (DataFrame): Dataset user top tracks
    """
    print("=" * 60)
    print("📂 LOADING & MERGING DATASET")
    print("=" * 60)

    # 1. Load dataset lagu (Spotify)
    lagu_path = os.path.join(DATASET_PATH, "lagu", "SpotifyFeatures.csv")
    lagu_df = pd.read_csv(lagu_path)
    print(f"✅ Spotify Features: {len(lagu_df):,} lagu")

    # 2. Load dataset lirik
    lirik_path = os.path.join(DATASET_PATH, "lirik", "lyrics-data.csv")
    lirik_df = pd.read_csv(lirik_path)
    print(f"✅ Song Lyrics: {len(lirik_df):,} lirik")

    # 3. Load dataset user
    user_path = os.path.join(DATASET_PATH, "user", "user_top_tracks.csv")
    user_df = pd.read_csv(user_path)
    print(f"✅ User Top Tracks: {len(user_df):,} user")

    # 4. Cleaning metadata
    print("\n🧹 Cleaning metadata...")
    lagu_df["clean_track"] = lagu_df["track_name"].apply(clean_text_metadata)
    lagu_df["clean_artist"] = lagu_df["artist_name"].apply(clean_text_metadata)

    lirik_df["artist_name_extracted"] = lirik_df["ALink"].apply(extract_artist_from_alink)
    lirik_df["clean_track"] = lirik_df["SName"].apply(clean_text_metadata)
    lirik_df["clean_artist"] = lirik_df["artist_name_extracted"].apply(clean_text_metadata)

    # 5. Merge (Inner Join)
    print("🔗 Merging lagu + lirik...")
    merged_df = pd.merge(
        lagu_df,
        lirik_df[["clean_track", "clean_artist", "Lyric"]],
        on=["clean_track", "clean_artist"],
        how="inner"
    )
    merged_df = merged_df.drop_duplicates(subset=['track_id'])

    # 6. Finalisasi
    merged_df = merged_df.sort_values("popularity", ascending=False).reset_index(drop=True)

    print(f"\n✅ Dataset final: {len(merged_df):,} lagu dengan lirik")
    print(f"📊 Popularitas: {merged_df['popularity'].min()} - {merged_df['popularity'].max()}")
    print(f"🎵 Kolom: {merged_df.columns.tolist()}")

    # 7. Simpan merged dataset
    if save_merged:
        merged_output = os.path.join(DATASET_PATH, "merged_songs.csv")
        merged_df.to_csv(merged_output, index=False)
        print(f"💾 Tersimpan di: {merged_output}")

    return merged_df, user_df


# =====================================================
# MAIN (jika dijalankan langsung)
# =====================================================
if __name__ == "__main__":
    merged_df, user_df = load_and_merge_dataset()
    print(f"\n📊 Contoh data:")
    print(merged_df[['track_name', 'artist_name', 'genre', 'popularity']].head(10))
