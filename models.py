"""
models.py — Fungsi untuk load model, generate embeddings, dan build FAISS index.

Modul ini berisi:
- load_model(): Load/download pre-trained word embedding model
- generate_tfidf_embeddings(): Generate TF-IDF weighted embeddings
- build_faiss_index(): Build FAISS index dari embeddings
- run_model_pipeline(): Pipeline lengkap (load → embed → index → evaluate)
"""

import os
import io
import gc
import tempfile
import shutil
import numpy as np
import faiss
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import FastText, KeyedVectors
import gensim.downloader as api

from config import BASE_PATH, MODELS_PATH, EMBEDDINGS_PATH, EMBEDDING_DIM, TOP_K
from preprocessing import clean_lyrics_universal
from evaluation import evaluate_kfold_genre_based


# =====================================================
# MODEL CONFIGURATIONS
# =====================================================
MODEL_CONFIGS = {
    'FastText': {
        'gensim_name': 'fasttext-wiki-news-subwords-300',
        'local_name': 'fasttext_wiki_news_300.kv',
        'embedding_name': 'fasttext_embeddings_tfidf.npy',
        'size_info': '~958 MB',
    },
    'GloVe': {
        'gensim_name': 'glove-wiki-gigaword-300',
        'local_name': 'glove_wiki_gigaword_300.kv',
        'embedding_name': 'glove_embeddings_tfidf.npy',
        'size_info': '~376 MB',
    },
    'Word2Vec': {
        'gensim_name': 'word2vec-google-news-300',
        'local_name': 'word2vec_google_news_300.kv',
        'embedding_name': 'word2vec_embeddings_tfidf.npy',
        'size_info': '~1.6 GB',
    },
}


# =====================================================
# FUNGSI LOAD MODEL
# =====================================================

def load_model(model_name):
    """
    Load pre-trained word embedding model (dari file lokal atau download).
    
    Args:
        model_name: 'FastText', 'GloVe', atau 'Word2Vec'
    
    Returns:
        model: KeyedVectors object
    """
    config = MODEL_CONFIGS[model_name]
    saved_path = os.path.join(MODELS_PATH, config['local_name'])

    model = None

    # Coba load dari file lokal
    if os.path.exists(saved_path):
        print(f"📂 Memuat {model_name} dari: {saved_path}")
        try:
            # Selalu pakai mmap='r' agar kompatibel dengan Google Drive Desktop
            # (file streaming/placeholder tidak bisa di-load langsung tanpa mmap)
            model = KeyedVectors.load(saved_path, mmap='r')
            print(f"✅ {model_name} loaded! Vocabulary: {len(model):,} kata")
        except Exception as e:
            print(f"⚠️ Gagal load: {e}")
            model = None

    # Download jika belum ada
    if model is None:
        print(f"📥 Downloading {model_name} ({config['gensim_name']}, {config['size_info']})...")
        try:
            model = api.load(config['gensim_name'])
            print(f"✅ Download selesai! Vocabulary: {len(model):,} kata")

            # Simpan ke Google Drive
            print(f"💾 Menyimpan ke: {saved_path}")
            with tempfile.TemporaryDirectory() as _tmpdir:
                _tmp_path = os.path.join(_tmpdir, config['local_name'])
                model.save(_tmp_path)
                os.makedirs(MODELS_PATH, exist_ok=True)
                for _f in os.listdir(_tmpdir):
                    shutil.copy2(os.path.join(_tmpdir, _f),
                                 os.path.join(MODELS_PATH, _f))
            print(f"✅ Model tersimpan di: {saved_path}")
        except Exception as e:
            print(f"❌ Gagal download: {e}")

    return model


# =====================================================
# FUNGSI GENERATE EMBEDDINGS
# =====================================================

def generate_tfidf_embeddings(model, merged_df, model_name):
    """
    Generate TF-IDF weighted embeddings untuk setiap lagu.
    Hasil disimpan sebagai .npy file.
    
    Args:
        model: KeyedVectors word embedding model
        merged_df: DataFrame dengan kolom 'Lyric'
        model_name: 'FastText', 'GloVe', atau 'Word2Vec'
    
    Returns:
        embeddings: numpy array (n_songs, 300)
    """
    config = MODEL_CONFIGS[model_name]
    emb_path = os.path.join(EMBEDDINGS_PATH, config['embedding_name'])

    # Cek apakah sudah ada
    if os.path.exists(emb_path):
        print(f"📂 Embeddings ditemukan, loading...")
        # Pakai mmap agar kompatibel dengan Google Drive Desktop
        embeddings = np.array(np.load(emb_path, mmap_mode='r')).astype('float32')
        print(f"✅ {model_name} Embeddings loaded: {embeddings.shape}")
    else:
        print(f"⚙️ Generating {model_name} embeddings dengan TF-IDF weighting...")

        # Bersihkan semua lirik
        cleaned_lyrics = [clean_lyrics_universal(str(t)) for t in merged_df['Lyric']]

        # Fit TF-IDF
        tfidf = TfidfVectorizer(max_features=50000, min_df=2)
        tfidf.fit(cleaned_lyrics)
        word_to_idf = {w: tfidf.idf_[i] for w, i in tfidf.vocabulary_.items()}

        # Generate weighted embeddings
        vectors = []
        for idx, text in enumerate(cleaned_lyrics):
            vec = np.zeros(EMBEDDING_DIM)
            if model and text:
                try:
                    words = text.split()
                    wvecs, wts = [], []
                    for word in words:
                        try:
                            wvec = model[word]
                            wvecs.append(wvec)
                            wts.append(word_to_idf.get(word, 1.0))
                        except KeyError:
                            pass
                    if wvecs:
                        vec = np.average(np.array(wvecs), axis=0,
                                         weights=np.array(wts))
                except Exception:
                    pass
            vectors.append(vec)
            if (idx + 1) % 5000 == 0:
                print(f"   Processed {idx+1}/{len(merged_df)}...")

        embeddings = np.array(vectors).astype('float32')
        faiss.normalize_L2(embeddings)

        # Simpan
        np.save(emb_path, embeddings)
        print(f"✅ {model_name} Embeddings tersimpan: {embeddings.shape}")

    faiss.normalize_L2(embeddings)
    return embeddings


# =====================================================
# FUNGSI BUILD FAISS INDEX
# =====================================================

def build_faiss_index(embeddings):
    """Build FAISS IndexFlatIP dari embeddings."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    print(f"✅ FAISS Index: {index.ntotal} vectors")
    return index


# =====================================================
# PIPELINE LENGKAP
# =====================================================

def run_model_pipeline(model_name, merged_df, free_model_memory=True):
    """
    Pipeline lengkap: load model → generate embeddings → evaluate.
    
    Args:
        model_name: 'FastText', 'GloVe', atau 'Word2Vec'
        merged_df: DataFrame hasil preprocessing
        free_model_memory: Hapus model dari memori setelah generate embeddings
    
    Returns:
        dict: {embeddings, index, results}
    """
    print(f"\n{'='*70}")
    print(f"🚀 PIPELINE: {model_name}")
    print(f"{'='*70}")

    # 1. Load model
    model = load_model(model_name)

    # 2. Generate embeddings
    embeddings = generate_tfidf_embeddings(model, merged_df, model_name)

    # 3. Free memory
    if free_model_memory and model is not None:
        del model
        gc.collect()
        print(f"🗑️ Model {model_name} dihapus dari memori")

    # 4. Build FAISS index
    index = build_faiss_index(embeddings)

    # 5. Evaluate
    print(f"\n📊 Mengevaluasi {model_name}...")
    results = evaluate_kfold_genre_based(
        embeddings=embeddings,
        merged_df=merged_df,
        top_k=TOP_K,
        use_related_genres=True
    )
    results['Model'] = model_name
    results['Training Method'] = 'pretrained'

    print(f"\n✅ {model_name} selesai!")
    print(f"   Hit Rate: {results['Hit Rate']:.2%}")
    print(f"   MRR: {results['MRR']:.4f}")
    print(f"   NDCG: {results['NDCG']:.4f}")

    return {
        'embeddings': embeddings,
        'index': index,
        'results': results
    }


# =====================================================
# MAIN (jika dijalankan langsung)
# =====================================================
if __name__ == "__main__":
    from preprocessing import load_and_merge_dataset

    merged_df, user_df = load_and_merge_dataset(save_merged=False)

    # Jalankan semua model
    all_results = {}
    for name in ['FastText', 'GloVe', 'Word2Vec']:
        all_results[name] = run_model_pipeline(name, merged_df)

    # Tampilkan leaderboard
    print(f"\n{'='*70}")
    print("🏆 LEADERBOARD")
    print(f"{'='*70}")
    for name, data in sorted(all_results.items(),
                              key=lambda x: x[1]['results']['NDCG'],
                              reverse=True):
        r = data['results']
        print(f"   {name}: Hit Rate={r['Hit Rate']:.2%}, "
              f"MRR={r['MRR']:.4f}, NDCG={r['NDCG']:.4f}")
