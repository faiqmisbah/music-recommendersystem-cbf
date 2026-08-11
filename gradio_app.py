"""
gradio_app.py — Modul User Interface (UI) interaktif untuk Sistem Rekomendasi Musik.
"""

import os
import numpy as np
import pandas as pd
import faiss
import gradio as gr

from config import EMBEDDINGS_PATH

# Dictionary global untuk data model dan performa
model_data = {}
model_hitrates = {'GloVe': 92.65, 'FastText': 90.83, 'Word2Vec': 90.44}
user_profile_df = None
merged_df = None

# Menentukan nama kolom dinamis
track_col = 'track_name'
artist_col = 'artist_name'
genre_col = 'genre'
lyric_col = 'lyrics'

def setup_gradio_data(df_input):
    """
    Menyiapkan data embeddings, FAISS indexes, user profiles, dan kolom metadata lagu.
    """
    global model_data, user_profile_df, merged_df
    global track_col, artist_col, genre_col, lyric_col

    merged_df = df_input
    
    print("\n📂 Loading embeddings untuk Gradio UI...")
    for name in ['GloVe', 'FastText', 'Word2Vec']:
        emb_path = os.path.join(EMBEDDINGS_PATH, f"{name.lower()}_embeddings_tfidf.npy")
        if name == 'Word2Vec':
            emb_path = os.path.join(EMBEDDINGS_PATH, "word2vec_embeddings_tfidf.npy")
            
        if os.path.exists(emb_path):
            emb = np.array(np.load(emb_path, mmap_mode='r')).astype('float32')
            emb_norm = emb.copy()
            faiss.normalize_L2(emb_norm)
            
            index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb_norm.astype('float32'))
            
            model_data[name] = {
                'embeddings': emb, 
                'embeddings_norm': emb_norm, 
                'index': index
            }
            print(f"   ✅ {name} loaded: {emb.shape}")
        else:
            print(f"   ⚠️ {name} embeddings not found at {emb_path}")

    if not model_data:
        raise ValueError("❌ Tidak ada embeddings yang berhasil di-load! Jalankan training model terlebih dahulu.")

    n_songs_valid = list(model_data.values())[0]['embeddings'].shape[0]

    # Generate user profiles (seed=42 untuk reproduktifitas)
    np.random.seed(42)
    list_user_profiles = []
    for i in range(100):
        n_liked = np.random.randint(5, 15)
        liked_songs = np.random.choice(n_songs_valid, size=n_liked, replace=False)
        play_counts = np.random.exponential(scale=30, size=n_liked).astype(int) + 5
        play_counts = np.clip(play_counts, 5, 150)
        play_counts = sorted(play_counts.tolist(), reverse=True)
        list_user_profiles.append({
            'user_id': i, 
            'liked_song_indices': liked_songs.tolist(), 
            'play_counts': play_counts
        })
    user_profile_df = pd.DataFrame(list_user_profiles)

    # Identifikasi kolom metadata lagu
    track_col = 'track_name' if 'track_name' in merged_df.columns else 'SName'
    artist_col = 'artist_name' if 'artist_name' in merged_df.columns else 'ALink'
    genre_col = 'genre' if 'genre' in merged_df.columns else None
    lyric_col = 'Lyric' if 'Lyric' in merged_df.columns else 'lyrics'

    print(f"✅ Gradio UI data siap: {n_songs_valid} lagu, {len(user_profile_df)} user profiles")


# ==============================================================================
# FUNGSI REKOMENDASI & UI INTERFACE
# ==============================================================================

def get_user_history(user_id):
    """Mendapatkan riwayat lagu favorit user beserta play count & lirik"""
    if user_profile_df is None or user_id < 0 or user_id >= len(user_profile_df):
        return None, "Invalid User ID!"

    user_data = user_profile_df.iloc[user_id]
    liked_indices = user_data['liked_song_indices']
    play_counts = user_data['play_counts']

    history_songs = []
    for i, idx in enumerate(liked_indices):
        if idx < len(merged_df):
            song = merged_df.iloc[idx]
            full_lyric = str(song.get(lyric_col, 'Lyrics not available'))

            history_songs.append({
                'index': idx,
                'track_name': song.get(track_col, 'Unknown'),
                'artist_name': song.get(artist_col, 'Unknown'),
                'genre': song.get(genre_col, 'Music') if genre_col else 'Music',
                'lyric_snippet': full_lyric[:220] + "..." if len(full_lyric) > 220 else full_lyric,
                'full_lyric': full_lyric,
                'play_count': play_counts[i] if i < len(play_counts) else 0
            })

    history_songs = sorted(history_songs, key=lambda x: x['play_count'], reverse=True)
    return history_songs, None


def get_recommendations(user_id, model_name, num_recommendations=10):
    """Menghasilkan lagu rekomendasi berdasarkan riwayat dan model embedding"""
    if user_profile_df is None or user_id < 0 or user_id >= len(user_profile_df):
        return None, "Invalid User ID!"

    if model_name not in model_data:
        return None, f"Model {model_name} is not available!"

    user_data = user_profile_df.iloc[user_id]
    liked_indices = user_data['liked_song_indices']

    embeddings = model_data[model_name]['embeddings']
    faiss_index = model_data[model_name]['index']

    user_embeddings = embeddings[liked_indices]
    user_vector = np.mean(user_embeddings, axis=0).reshape(1, -1).astype('float32')
    faiss.normalize_L2(user_vector)

    k = num_recommendations + len(liked_indices)
    distances, indices = faiss_index.search(user_vector, k)

    recommendations = []
    for i, idx in enumerate(indices[0]):
        if idx not in liked_indices and idx < len(merged_df):
            song = merged_df.iloc[idx]
            full_lyric = str(song.get(lyric_col, 'Lyrics not available'))

            recommendations.append({
                'index': idx,
                'track_name': song.get(track_col, 'Unknown'),
                'artist_name': song.get(artist_col, 'Unknown'),
                'genre': song.get(genre_col, 'Music') if genre_col else 'Music',
                'lyric_snippet': full_lyric[:220] + "..." if len(full_lyric) > 220 else full_lyric,
                'similarity': float(distances[0][i])
            })

        if len(recommendations) >= num_recommendations:
            break

    return recommendations, None


def display_user_info(user_id_str, num_history_results=10):
    """Merender data riwayat pendengar ke format HTML (2-Column Side-by-Side per Lagu)"""
    try:
        user_id = int(user_id_str)
        num_history = int(num_history_results)
    except:
        return "<p style='color:#ef4444; font-size:16px; font-weight:bold; padding:14px; background:#450a0a; border-radius:10px;'>Please enter a valid User ID (number 0–99)</p>", ""

    history, error = get_user_history(user_id)
    if error:
        return f"<p style='color:#ef4444; font-size:16px; font-weight:bold; padding:14px; background:#450a0a; border-radius:10px;'>{error}</p>", ""

    total_plays = sum(song['play_count'] for song in history)
    display_history = history[:num_history]

    header_html = f"""
    <div style="padding: 14px 20px; background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); 
                border-radius: 12px; margin-bottom: 16px; box-shadow: 0 4px 16px rgba(79, 70, 229, 0.25); display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
        <span style="color: #ffffff; font-size: 18px; font-weight: 800;">User ID: {user_id}</span>
        <span style="color: #e0e7ff; font-size: 15px; font-weight: 600;">
            Total Liked Songs: <b style="color:#ffffff; font-size:16px;">{len(history)}</b> &nbsp;&bull;&nbsp; 
            Total Plays: <b style="color:#ffffff; font-size:16px;">{total_plays:,}</b>
        </span>
    </div>
    <h3 style="color:#f8fafc; font-size: 18px; font-weight: 800; margin-bottom: 14px;">
        Top {len(display_history)} Songs Liked by This User:
    </h3>
    """

    history_html = ""
    for i, song in enumerate(display_history):
        if song['play_count'] >= 100:
            play_badge_bg = "#7f1d1d"
            play_badge_text = "#fca5a5"
            play_badge_border = "#ef4444"
            play_status = "High Frequency"
        elif song['play_count'] >= 50:
            play_badge_bg = "#78350f"
            play_badge_text = "#fcd34d"
            play_badge_border = "#f59e0b"
            play_status = "Medium Frequency"
        else:
            play_badge_bg = "#1e3a8a"
            play_badge_text = "#93c5fd"
            play_badge_border = "#3b82f6"
            play_status = "Standard"

        if i == 0:
            badge_color = "linear-gradient(135deg, #f59e0b, #d97706)"
            badge_text = "#1 Rank"
        elif i == 1:
            badge_color = "linear-gradient(135deg, #94a3b8, #64748b)"
            badge_text = "#2 Rank"
        elif i == 2:
            badge_color = "linear-gradient(135deg, #b45309, #78350f)"
            badge_text = "#3 Rank"
        else:
            badge_color = "linear-gradient(135deg, #4f46e5, #4338ca)"
            badge_text = f"#{i+1} Rank"

        history_html += f"""
        <div style="border: 1px solid rgba(255, 255, 255, 0.08); padding: 18px 22px; margin-bottom: 14px; border-radius: 14px; 
                    background: #1e293b; box-shadow: 0 4px 16px rgba(0,0,0,0.25);">
            <div style="display: flex; gap: 20px; align-items: flex-start; justify-content: space-between; flex-wrap: wrap;">
                <!-- Left Column: Metadata (~40% width) -->
                <div style="flex: 1; min-width: 220px; display: flex; flex-direction: column; gap: 8px;">
                    <div>
                        <span style="background:{badge_color}; color:#ffffff; padding: 5px 14px; border-radius: 16px; 
                                     font-weight: 800; font-size: 14px;">{badge_text}</span>
                    </div>
                    <div style="font-weight: 800; font-size: 21px; color: #f8fafc; letter-spacing: -0.3px; line-height: 1.25;">
                        {song['track_name']}
                    </div>
                    <div style="color: #94a3b8; font-size: 15px; font-weight: 600; display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                        <span>Artist: <b style="color: #cbd5e1;">{song['artist_name']}</b></span>
                        <span style="color: #64748b;">&bull;</span>
                        <span style="background: rgba(255,255,255,0.06); padding: 3px 10px; border-radius: 6px; color: #a7f3d0; font-size: 14px; font-weight: 600;">Genre: {song['genre']}</span>
                    </div>
                    <div>
                        <span style="background:{play_badge_bg}; color:{play_badge_text}; border: 1px solid {play_badge_border}55; 
                                     padding: 5px 14px; border-radius: 16px; font-size: 14px; font-weight: 700; display: inline-block;">
                            {song['play_count']:,} plays ({play_status})
                        </span>
                    </div>
                </div>
                
                <!-- Right Column: Lyrics Snippet (Wider ~60% width) -->
                <div style="flex: 1.5; min-width: 260px; background: #0f172a; padding: 14px 18px; border-radius: 12px; 
                            border-left: 4px solid #6366f1; border: 1px solid rgba(255,255,255,0.05); border-left-width: 4px;">
                    <div style="font-size: 12px; color: #818cf8; margin-bottom: 6px; font-weight: 800; letter-spacing: 0.5px;">
                        LYRICS SNIPPET:
                    </div>
                    <div style="font-style: italic; font-size: 15px; color: #e2e8f0; line-height: 1.6; font-weight: 400;">
                        "{song['lyric_snippet']}"
                    </div>
                </div>
            </div>
        </div>
        """

    if len(history) > num_history:
        history_html += f"""
        <p style="color:#94a3b8; text-align:center; font-size:14px; font-weight:600; padding:8px;">
            ... and {len(history) - num_history} more songs in history
        </p>
        """

    return header_html + history_html, str(user_id)


def generate_recommendations_html(user_id_str, model_name, num_results):
    """Merender rekomendasi musik ke format HTML (2-Column Side-by-Side per Lagu)"""
    try:
        user_id = int(user_id_str)
    except:
        return "<p style='color:#ef4444; font-size:16px; font-weight:bold; padding:14px; background:#450a0a; border-radius:10px;'>Please enter a valid User ID!</p>"

    recommendations, error = get_recommendations(user_id, model_name, int(num_results))
    if error:
        return f"<p style='color:#ef4444; font-size:16px; font-weight:bold; padding:14px; background:#450a0a; border-radius:10px;'>{error}</p>"

    model_gradients = {
        'GloVe': ('#059669', '#10b981', '#10b981'),
        'FastText': ('#4f46e5', '#6366f1', '#6366f1'),
        'Word2Vec': ('#e11d48', '#f43f5e', '#f43f5e')
    }
    grad_start, grad_end, model_color = model_gradients.get(model_name, ('#4f46e5', '#7c3aed', '#6366f1'))
    hit_rate = model_hitrates.get(model_name, 0)
    n_liked = len(user_profile_df.iloc[user_id]['liked_song_indices'])

    result_html = f"""
    <div style="padding: 14px 20px; background: linear-gradient(135deg, {grad_start} 0%, {grad_end} 100%); 
                border-radius: 12px; margin-bottom: 16px; box-shadow: 0 4px 16px {grad_start}33; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
        <span style="color: #ffffff; font-size: 18px; font-weight: 800;">Recommendations for User {user_id}</span>
        <span style="color: #e0e7ff; font-size: 15px; font-weight: 600;">
            Model: <b style="color:#ffffff; font-size:16px;">{model_name}</b> &nbsp;&bull;&nbsp; 
            Hit Rate: <b style="color:#ffffff; font-size:16px;">{hit_rate}%</b> &nbsp;&bull;&nbsp; 
            Based on <b style="color:#ffffff; font-size:16px;">{n_liked}</b> songs
        </span>
    </div>
    """

    for rank, song in enumerate(recommendations):
        if rank == 0:
            badge_color = "linear-gradient(135deg, #f59e0b, #d97706)"
            badge_text = "#1 Rank"
        elif rank == 1:
            badge_color = "linear-gradient(135deg, #94a3b8, #64748b)"
            badge_text = "#2 Rank"
        elif rank == 2:
            badge_color = "linear-gradient(135deg, #b45309, #78350f)"
            badge_text = "#3 Rank"
        else:
            badge_color = f"linear-gradient(135deg, {grad_start}, {grad_end})"
            badge_text = f"#{rank+1} Rank"

        result_html += f"""
        <div style="border: 1px solid rgba(255, 255, 255, 0.08); padding: 18px 22px; margin-bottom: 14px; border-radius: 14px; 
                    background: #1e293b; box-shadow: 0 4px 16px rgba(0,0,0,0.25); border-left: 5px solid {model_color};">
            <div style="display: flex; gap: 20px; align-items: flex-start; justify-content: space-between; flex-wrap: wrap;">
                <!-- Left Column: Metadata (~40% width) -->
                <div style="flex: 1; min-width: 220px; display: flex; flex-direction: column; gap: 8px;">
                    <div>
                        <span style="background:{badge_color}; color:#ffffff; padding: 5px 14px; border-radius: 16px; 
                                     font-weight: 800; font-size: 14px;">{badge_text}</span>
                    </div>
                    <div style="font-weight: 800; font-size: 21px; color: #f8fafc; letter-spacing: -0.3px; line-height: 1.25;">
                        {song['track_name']}
                    </div>
                    <div style="color: #94a3b8; font-size: 15px; font-weight: 600; display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                        <span>Artist: <b style="color: #cbd5e1;">{song['artist_name']}</b></span>
                        <span style="color: #64748b;">&bull;</span>
                        <span style="background: rgba(255,255,255,0.06); padding: 3px 10px; border-radius: 6px; color: #a7f3d0; font-size: 14px; font-weight: 600;">Genre: {song['genre']}</span>
                    </div>
                    <div>
                        <span style="background: #064e3b; color: #34d399; border: 1px solid #059669; 
                                     padding: 5px 14px; border-radius: 16px; font-size: 14px; font-weight: 800; display: inline-block;">
                            Similarity: {song['similarity']:.4f}
                        </span>
                    </div>
                </div>
                
                <!-- Right Column: Lyrics Snippet (Wider ~60% width) -->
                <div style="flex: 1.5; min-width: 260px; background: #0f172a; padding: 14px 18px; border-radius: 12px; 
                            border-left: 4px solid {model_color}; border: 1px solid rgba(255,255,255,0.05); border-left-width: 4px;">
                    <div style="font-size: 12px; color: #818cf8; margin-bottom: 6px; font-weight: 800; letter-spacing: 0.5px;">
                        LYRICS SNIPPET:
                    </div>
                    <div style="font-style: italic; font-size: 15px; color: #e2e8f0; line-height: 1.6; font-weight: 400;">
                        "{song['lyric_snippet']}"
                    </div>
                </div>
            </div>
        </div>
        """

    return result_html


# Custom CSS Theme untuk Gradio Blocks
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

body, .gradio-container {
    background-color: #0b0f19 !important;
    color: #f8fafc !important;
}

/* Master Step Cards */
.step-card {
    background: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    padding: 16px 18px !important;
    margin-bottom: 6px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
    height: 100% !important;
}

/* Remove all nested inner boxes inside step-cards */
.step-card .gr-block, 
.step-card .gr-box, 
.step-card .block, 
.step-card .gr-form,
.step-card .form,
.step-card fieldset,
.step-card .gr-panel,
.step-card .row {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Step Card Title Styling */
.step-title {
    color: #818cf8 !important;
    font-size: 14px !important;
    font-weight: 800 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    margin-bottom: 10px !important;
    display: block !important;
}

/* Custom styling untuk Labels, Inputs, Dropdowns */
label, .gr-form label, span.text-gray-500, label span {
    font-size: 14px !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
    margin-bottom: 4px !important;
}

.gr-input, .gr-select, select, input, textarea {
    background-color: #0f172a !important;
    color: #f8fafc !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 8px 12px !important;
}

.gr-input:focus, .gr-select:focus, select:focus, input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.3) !important;
}

/* Fix Gradio Slider Reset Icon & Formatting Glitch */
.gr-slider, .gr-form .gr-slider {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

.gr-slider .head, label .head {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    width: 100% !important;
    margin-bottom: 4px !important;
}

/* Completely hide ugly reset button inside slider head */
.gr-slider button,
.gr-slider-reset,
.gr-slider .head button,
.gr-slider .head svg,
.gr-slider .head input+div,
.gr-slider .head button+div,
.gr-slider .head .gr-button {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    opacity: 0 !important;
    visibility: hidden !important;
    position: absolute !important;
    left: -9999px !important;
}

.gr-slider input[type="number"], 
.gr-form input[type="number"],
input[type="number"].gr-input {
    width: 42px !important;
    min-width: 42px !important;
    max-width: 42px !important;
    height: 24px !important;
    padding: 0 !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    text-align: center !important;
    background: #0f172a !important;
    color: #818cf8 !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    border-radius: 4px !important;
    box-shadow: none !important;
}

input[type=range] {
    accent-color: #6366f1 !important;
    height: 6px !important;
}

/* Buttons inside Cards */
.step-card button, .step-card .lg, button {
    width: 100% !important;
    margin-top: 10px !important;
    border-radius: 8px !important;
    padding: 10px 16px !important;
    font-size: 15px !important;
    font-weight: 800 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

button.secondary {
    background: linear-gradient(135deg, #334155 0%, #1e293b 100%) !important;
    color: #f8fafc !important;
    border: 1px solid #475569 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
}

button.secondary:hover {
    background: linear-gradient(135deg, #475569 0%, #334155 100%) !important;
    border-color: #818cf8 !important;
    transform: translateY(-1px) !important;
}

button.primary {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.45) !important;
}

button.primary:hover {
    background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%) !important;
    box-shadow: 0 6px 22px rgba(99, 102, 241, 0.65) !important;
    transform: translateY(-1px) !important;
}

hr {
    border-color: rgba(255, 255, 255, 0.08) !important;
}
"""


def build_gradio_interface():
    """
    Membangun instance UI Gradio (Gradio Blocks).
    """
    available_models = list(model_data.keys())
    available_users = list(range(len(user_profile_df)))

    with gr.Blocks(
        title="Music Recommendation System",
        css=CUSTOM_CSS,
        theme=gr.themes.Base()
    ) as demo:

        # Header Banner
        gr.HTML("""
            <div style="text-align:center; padding: 18px 22px; background: linear-gradient(135deg, #1e1b4b 0%, #311042 50%, #1e293b 100%); 
                        border-radius: 14px; border: 1px solid rgba(139, 92, 246, 0.3); margin-bottom: 14px; 
                        box-shadow: 0 4px 18px rgba(99, 102, 241, 0.2);">
                <h1 style="color: #ffffff; font-size: 2.1em; font-weight: 800; margin: 0 0 4px 0; letter-spacing: -0.5px;">
                    Music Recommendation System
                </h1>
                <p style="color: #cbd5e1; font-size: 1.05em; font-weight: 500; margin: 0;">
                    Viewing User History and Getting Music Recommendations
                </p>
            </div>
        """)

        # TOP SECTION: Step 1, Step 2, Step 3 Side-by-Side horizontally!
        with gr.Row():
            # Step 1: Configuration (2 Columns Side-by-Side inside, Vertical stack on right)
            with gr.Column(scale=1, elem_classes=["step-card"]):
                gr.HTML("<span class='step-title'>Step 1: Configuration</span>")
                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        model_input = gr.Dropdown(
                            choices=available_models,
                            value=available_models[0] if available_models else "GloVe",
                            label="Select Model"
                        )
                    with gr.Column(scale=1, min_width=120):
                        model_info_html = gr.HTML("""
                            <div style="background: rgba(16, 185, 129, 0.12); padding: 8px 12px; border-radius: 8px; margin-top: 18px; border: 1px solid rgba(16, 185, 129, 0.3); border-left: 4px solid #10b981; display:flex; flex-direction:column; gap:3px; justify-content:center; box-sizing: border-box;">
                                <div style="font-size: 13px; color: #f8fafc; font-weight: 600;">Model: <b style="color:#ffffff;">GloVe</b></div>
                                <div style="font-size: 13px; color: #34d399; font-weight: 800;">Hit Rate: <b style="color:#34d399;">92.65%</b></div>
                            </div>
                        """)

            # Step 2: Select User & History Count (2 Columns Side-by-Side inside)
            with gr.Column(scale=1, elem_classes=["step-card"]):
                gr.HTML("<span class='step-title'>Step 2: Select User</span>")
                with gr.Row():
                    with gr.Column(scale=1, min_width=80):
                        user_id_input = gr.Dropdown(
                            choices=[str(i) for i in available_users],
                            value="0",
                            label="User ID"
                        )
                    with gr.Column(scale=1, min_width=120):
                        num_history_results = gr.Slider(
                            minimum=5,
                            maximum=20,
                            value=10,
                            step=1,
                            label="History Songs"
                        )
                view_history_btn = gr.Button("View User History", variant="secondary", size="lg")

            # Step 3: Get Recommendations
            with gr.Column(scale=1, elem_classes=["step-card"]):
                gr.HTML("<span class='step-title'>Step 3: Get Recommendations</span>")
                num_results = gr.Slider(
                    minimum=5,
                    maximum=20,
                    value=10,
                    step=1,
                    label="Number of Recommendations"
                )
                recommend_btn = gr.Button("Get Recommendations", variant="primary", size="lg")

        current_user_state = gr.State("")

        # MIDDLE SECTION: Outputs side-by-side (2 Columns)
        with gr.Row():
            # Left Column: User History
            with gr.Column(scale=1):
                gr.HTML("<h3 style='color:#f8fafc; font-size:18px; font-weight:800; margin-top:2px; margin-bottom:8px;'>User History</h3>")
                history_output = gr.HTML(
                    value="<div style='background:#1e293b; border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:36px; text-align:center; color:#94a3b8; font-size:15px; font-weight:600;'>Select a User ID and click 'View User History'...</div>"
                )

            # Right Column: Music Recommendations
            with gr.Column(scale=1):
                gr.HTML("<h3 style='color:#f8fafc; font-size:18px; font-weight:800; margin-top:2px; margin-bottom:8px;'>Music Recommendations</h3>")
                recommend_output = gr.HTML(
                    value="<div style='background:#1e293b; border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:36px; text-align:center; color:#94a3b8; font-size:15px; font-weight:600;'>Recommendations will appear here after clicking 'Get Recommendations'...</div>"
                )

        # BOTTOM SECTION: Model Ranking & Usage Instructions
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                    <div style="background: #1e293b; padding: 16px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); height: 100%;">
                        <h4 style="margin: 0 0 12px 0; color: #f8fafc; font-size: 15px; font-weight: 800;">
                            Model Performance Ranking
                        </h4>
                        <div style="display:flex; gap:10px; flex-wrap:wrap;">
                            <div style="flex:1; min-width:140px; background: #0f172a; padding: 10px 14px; border-radius: 8px; border-left: 4px solid #10b981; display:flex; justify-content:space-between; align-items:center;">
                                <span style="color: #ffffff; font-size: 13px; font-weight: 700;">Rank #1: GloVe <span style="font-size:10px; background:#065f46; color:#6ee7b7; padding:2px 5px; border-radius:4px; margin-left:3px;">BEST</span></span>
                                <span style="color: #34d399; font-size: 13px; font-weight: 800;">92.65%</span>
                            </div>
                            <div style="flex:1; min-width:140px; background: #0f172a; padding: 10px 14px; border-radius: 8px; border-left: 4px solid #6366f1; display:flex; justify-content:space-between; align-items:center;">
                                <span style="color: #ffffff; font-size: 13px; font-weight: 600;">Rank #2: FastText</span>
                                <span style="color: #818cf8; font-size: 13px; font-weight: 700;">90.83%</span>
                            </div>
                            <div style="flex:1; min-width:140px; background: #0f172a; padding: 10px 14px; border-radius: 8px; border-left: 4px solid #f43f5e; display:flex; justify-content:space-between; align-items:center;">
                                <span style="color: #ffffff; font-size: 13px; font-weight: 600;">Rank #3: Word2Vec</span>
                                <span style="color: #fb7185; font-size: 13px; font-weight: 700;">90.44%</span>
                            </div>
                        </div>
                    </div>
                """)
            with gr.Column(scale=1):
                gr.HTML("""
                    <div style="background: rgba(30, 41, 59, 0.8); padding: 16px 18px; border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.3); border-left: 4px solid #6366f1; height: 100%;">
                        <h4 style="margin: 0 0 8px 0; color: #818cf8; font-size: 15px; font-weight: 800;">
                            How to Use
                        </h4>
                        <ol style="margin: 0; padding-left: 18px; color: #cbd5e1; font-size: 13.5px; line-height: 1.6; font-weight: 500;">
                            <li>Select a <b style="color:#ffffff;">Model</b> under Step 1</li>
                            <li>Select a <b style="color:#ffffff;">User ID</b> &amp; count under Step 2</li>
                            <li>Click <b style="color:#818cf8;">"View User History"</b></li>
                            <li>Set recommendations &amp; click <b style="color:#a78bfa;">"Get Recommendations"</b></li>
                        </ol>
                    </div>
                """)

        # Event handlers
        def update_model_info(model_name):
            colors = {'GloVe': '#10b981', 'FastText': '#6366f1', 'Word2Vec': '#f43f5e'}
            hit_rate = model_hitrates.get(model_name, 0)
            color = colors.get(model_name, '#6366f1')
            return f"""
                <div style="background: {color}18; padding: 8px 12px; border-radius: 8px; margin-top: 18px; border: 1px solid {color}44; border-left: 4px solid {color}; display:flex; flex-direction:column; gap:3px; justify-content:center; box-sizing: border-box;">
                    <div style="font-size: 13px; color: #f8fafc; font-weight: 600;">Model: <b style="color:#ffffff;">{model_name}</b></div>
                    <div style="font-size: 13px; color: {color}; font-weight: 800;">Hit Rate: <b style="color:{color};">{hit_rate}%</b></div>
                </div>
            """

        model_input.change(
            fn=update_model_info,
            inputs=[model_input],
            outputs=[model_info_html]
        )

        view_history_btn.click(
            fn=display_user_info,
            inputs=[user_id_input, num_history_results],
            outputs=[history_output, current_user_state]
        )

        recommend_btn.click(
            fn=generate_recommendations_html,
            inputs=[user_id_input, model_input, num_results],
            outputs=recommend_output
        )

        gr.HTML("""
            <div style="text-align:center; padding:18px; margin-top:28px; border-top:1px solid rgba(255,255,255,0.08);">
                <p style="color:#94a3b8; font-size:13px; line-height:1.6; font-weight:500;">
                    <b>Music Recommendation System Based on User Listening History</b><br>
                    GloVe Word Embedding with TF-IDF Weighting &amp; Content-Based Filtering<br>
                    © 2025 — Faiq Misbah Yazdi — Final Project (Tugas Akhir)
                </p>
            </div>
        """)

    return demo


def launch_gradio_app(df_input, share=True):
    """
    Fungsi utama untuk di-import dari notebook: menyiapkan data dan meluncurkan Gradio UI.
    """
    setup_gradio_data(df_input)
    demo = build_gradio_interface()
    
    print("\n🚀 Menjalankan Gradio UI...")
    print("=" * 80)
    demo.launch(share=share)
