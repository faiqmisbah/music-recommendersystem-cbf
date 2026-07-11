"""
evaluation.py — Fungsi evaluasi K-Fold Cross Validation.

Modul ini berisi:
- get_related_genres(): Mendapatkan genre yang berhubungan
- evaluate_model_kfold(): Evaluasi Leave-One-Out berbasis user history
- evaluate_kfold_genre_based(): Evaluasi K-Fold CV berbasis genre (UTAMA)
"""

import numpy as np
import faiss
from sklearn.model_selection import KFold
from config import GENRE_GROUPS, K_FOLDS, TOP_K, RANDOM_SEED


def get_related_genres(genre):
    """Mendapatkan set genre yang related/berhubungan."""
    for group_name, genres in GENRE_GROUPS.items():
        if genre in genres:
            return set(genres)
    return {genre}


def evaluate_model_kfold(embeddings, index, merged_df, user_df, k_splits=5, n_users=100, top_k=10):
    """
    Melakukan Cross-Validation berbasis user history (Leave-One-Out).
    Metode: Satu lagu disembunyikan sebagai target, sisanya jadi profil.
    
    (Fungsi ini dari cell 19 original notebook — disertakan untuk kelengkapan)
    """
    kf = KFold(n_splits=k_splits, shuffle=True, random_state=42)

    fold_hit_rates = []
    fold_mrrs = []
    fold_ndcgs = []

    user_histories = user_df['liked_song_indices'].tolist()
    all_user_ids = np.arange(len(user_df))

    print(f"📊 Memulai Evaluasi {k_splits}-Fold pada {len(user_df)} user...")

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(all_user_ids)):
        hits = []
        reciprocal_ranks = []
        ndcgs = []

        for u_id in test_idx:
            history_indices = user_histories[u_id]

            if len(history_indices) < 2:
                continue

            target_song_index = history_indices[-1]
            context_song_indices = history_indices[:-1]

            context_vectors = embeddings[context_song_indices]
            user_vector = np.mean(context_vectors, axis=0).reshape(1, -1).astype('float32')
            faiss.normalize_L2(user_vector)

            distances, predictions = index.search(user_vector, top_k + len(history_indices))
            prediction_list = predictions[0]

            final_recommendations = []
            for song_idx in prediction_list:
                if song_idx not in context_song_indices:
                    final_recommendations.append(song_idx)
                if len(final_recommendations) == top_k:
                    break

            if target_song_index in final_recommendations:
                hits.append(1)
                rank = final_recommendations.index(target_song_index) + 1
                reciprocal_ranks.append(1 / rank)
                ndcgs.append(1 / np.log2(rank + 1))
            else:
                hits.append(0)
                reciprocal_ranks.append(0)
                ndcgs.append(0)

        fold_hit = np.mean(hits) if hits else 0
        fold_mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
        fold_ndcg = np.mean(ndcgs) if ndcgs else 0

        fold_hit_rates.append(fold_hit)
        fold_mrrs.append(fold_mrr)
        fold_ndcgs.append(fold_ndcg)

    avg_hit_rate = np.mean(fold_hit_rates)
    avg_mrr = np.mean(fold_mrrs)
    avg_ndcg = np.mean(fold_ndcgs)

    best_fold_idx = np.argmax(fold_ndcgs)

    print(f"✅ Evaluasi Selesai. Rata-rata Hit Rate: {avg_hit_rate:.2%}")

    return {
        "Hit Rate": avg_hit_rate,
        "MRR": avg_mrr,
        "NDCG": avg_ndcg,
        "Best Split": f"Fold {best_fold_idx+1}"
    }


def evaluate_kfold_genre_based(embeddings, merged_df, k_folds=K_FOLDS,
                                top_k=TOP_K, use_related_genres=True):
    """
    K-Fold Cross Validation untuk evaluasi rekomendasi berbasis Genre.
    
    Optimisasi:
    - TF-IDF weighted embeddings
    - Top-K = 20 (standar banyak paper)
    - Related genres dianggap relevan (partial match)
    
    Returns: dict dengan hasil dan DataFrame per fold
    """
    if 'genre' not in merged_df.columns:
        print("⚠️ Kolom 'genre' tidak ditemukan!")
        return {"Hit Rate": 0, "MRR": 0, "NDCG": 0}

    idx_to_genre = merged_df['genre'].to_dict()

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_SEED)
    all_indices = np.arange(len(merged_df))

    fold_results = []

    print(f"   Menggunakan {k_folds}-Fold Cross Validation...")
    print(f"   Top-K: {top_k}")
    print(f"   Related Genres: {'Ya' if use_related_genres else 'Tidak'}")
    print()

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(all_indices)):
        # Buat FAISS index dari training set saja
        train_embeddings = embeddings[train_idx].astype('float32')

        fold_index = faiss.IndexFlatIP(train_embeddings.shape[1])
        fold_index.add(train_embeddings)

        # Evaluasi pada test set
        hits = []
        mrrs = []
        ndcgs = []

        for query_idx in test_idx:
            query_genre = idx_to_genre[query_idx]
            query_vec = embeddings[query_idx].reshape(1, -1).astype('float32')

            # Get related genres jika diaktifkan
            if use_related_genres:
                relevant_genres = get_related_genres(query_genre)
            else:
                relevant_genres = {query_genre}

            # Search di training index
            distances, predictions = fold_index.search(query_vec, top_k)

            # Map prediksi kembali ke index asli
            recommended_indices = [train_idx[p] for p in predictions[0]]

            # Binary relevance (1 jika genre sama atau related)
            relevance = [1 if idx_to_genre.get(rec_idx) in relevant_genres else 0
                        for rec_idx in recommended_indices]

            # Hit Rate
            hit = 1 if sum(relevance) > 0 else 0
            hits.append(hit)

            # MRR
            if hit:
                first_hit_rank = relevance.index(1) + 1
                mrrs.append(1 / first_hit_rank)
            else:
                mrrs.append(0)

            # NDCG
            dcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(relevance)])
            ideal_relevance = sorted(relevance, reverse=True)
            idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance)])
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs.append(ndcg)

        fold_hit = np.mean(hits)
        fold_mrr = np.mean(mrrs)
        fold_ndcg = np.mean(ndcgs)

        # Simpan hasil per fold
        fold_results.append({
            'Fold': f'Fold {fold_idx + 1}',
            'Hit Rate (%)': fold_hit * 100,
            'MRR': fold_mrr,
            'NDCG': fold_ndcg,
            'Test Samples': len(test_idx)
        })

    # Hitung rata-rata dan std
    hit_rates = [r['Hit Rate (%)'] for r in fold_results]
    mrrs_all = [r['MRR'] for r in fold_results]
    ndcgs_all = [r['NDCG'] for r in fold_results]

    avg_hit_rate = np.mean(hit_rates)
    avg_mrr = np.mean(mrrs_all)
    avg_ndcg = np.mean(ndcgs_all)

    std_hit = np.std(hit_rates)
    std_mrr = np.std(mrrs_all)
    std_ndcg = np.std(ndcgs_all)

    # Tambahkan baris rata-rata (sama persis dengan original)
    fold_results.append({
        'Fold': '📊 Rata-rata',
        'Hit Rate (%)': avg_hit_rate,
        'MRR': avg_mrr,
        'NDCG': avg_ndcg,
        'Test Samples': sum([r['Test Samples'] for r in fold_results[:-1]
                             if 'Test Samples' in r]) // k_folds
    })

    # Tambahkan baris std deviation
    fold_results.append({
        'Fold': '📈 Std Dev',
        'Hit Rate (%)': std_hit,
        'MRR': std_mrr,
        'NDCG': std_ndcg,
        'Test Samples': '-'
    })

    return {
        "Hit Rate": avg_hit_rate / 100,  # Konversi kembali ke decimal
        "MRR": avg_mrr,
        "NDCG": avg_ndcg,
        "Hit Rate Std": std_hit / 100,
        "MRR Std": std_mrr,
        "NDCG Std": std_ndcg,
        "K-Folds": k_folds,
        "Top-K": top_k,
        "Fold Results": fold_results
    }
