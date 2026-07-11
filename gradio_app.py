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
            # Pakai mmap_mode='r' agar kompatibel dengan Google Drive Desktop
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
                'lyric_snippet': full_lyric[:200] + "..." if len(full_lyric) > 200 else full_lyric,
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
                'lyric_snippet': full_lyric[:200] + "..." if len(full_lyric) > 200 else full_lyric,
                'similarity': float(distances[0][i])
            })

        if len(recommendations) >= num_recommendations:
            break

    return recommendations, None


def display_user_info(user_id_str):
    """Merender data riwayat pendengar ke format HTML"""
    try:
        user_id = int(user_id_str)
    except:
        return "<p style='color:red;'>⚠️ Please enter a valid User ID (number 0–99)</p>", ""

    history, error = get_user_history(user_id)
    if error:
        return f"<p style='color:red;'>⚠️ {error}</p>", ""

    total_plays = sum(song['play_count'] for song in history)

    header_html = f"""
    <div style="padding:20px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius:15px; margin-bottom:20px;">
        <h2 style="color:white; margin:0;">👤 User ID: {user_id}</h2>
        <p style="color:#e0e0e0; margin:5px 0 0 0;">
            Total liked songs: <b>{len(history)}</b> &nbsp;|&nbsp; 
            Total plays: <b>{total_plays:,}</b>
        </p>
    </div>
    <h3 style="margin-bottom:15px;">📚 Top 10 Songs Liked by This User:</h3>
    """

    history_html = ""
    for i, song in enumerate(history[:10]):
        if song['play_count'] >= 100:
            play_badge_color = "#e74c3c"
            play_icon = "🔥"
        elif song['play_count'] >= 50:
            play_badge_color = "#f39c12"
            play_icon = "⭐"
        else:
            play_badge_color = "#3498db"
            play_icon = "🎧"

        if i == 0:
            badge_color = "#FFD700"
            badge_text = "🥇 #1"
        elif i == 1:
            badge_color = "#C0C0C0"
            badge_text = "🥈 #2"
        elif i == 2:
            badge_color = "#CD7F32"
            badge_text = "🥉 #3"
        else:
            badge_color = "#667eea"
            badge_text = f"#{i+1}"

        history_html += f"""
        <div style="border:1px solid #e0e0e0; padding:15px; margin-bottom:15px; border-radius:12px; 
                    background:white; box-shadow:0 2px 8px rgba(0,0,0,0.1);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <span style="background:{badge_color}; color:white; padding:4px 12px; border-radius:20px; 
                             font-weight:bold; font-size:12px;">{badge_text}</span>
                <span style="background:{play_badge_color}; color:white; padding:4px 12px; border-radius:20px; 
                             font-size:12px; font-weight:bold;">
                    {play_icon} {song['play_count']:,}x played
                </span>
            </div>
            <div style="font-weight:bold; font-size:18px; color:#333; margin-bottom:5px;">
                🎵 {song['track_name']}
            </div>
            <div style="color:#666; font-size:14px; margin-bottom:10px;">
                👤 {song['artist_name']} &nbsp;|&nbsp; 🎸 {song['genre']}
            </div>
            <div style="background:#f5f5f5; padding:12px; border-radius:8px; 
                        border-left:4px solid #667eea;">
                <div style="font-size:11px; color:#888; margin-bottom:5px; font-weight:bold;">
                    📝 LYRICS:
                </div>
                <div style="font-style:italic; font-size:13px; color:#555; line-height:1.5;">
                    "{song['lyric_snippet']}"
                </div>
            </div>
        </div>
        """

    if len(history) > 10:
        history_html += f"""
        <p style="color:#888; text-align:center; font-size:13px;">
            ... and {len(history) - 10} more songs
        </p>
        """

    return header_html + history_html, str(user_id)


def generate_recommendations_html(user_id_str, model_name, num_results):
    """Merender rekomendasi musik ke format HTML"""
    try:
        user_id = int(user_id_str)
    except:
        return "<p style='color:red;'>⚠️ Please enter a valid User ID!</p>"

    recommendations, error = get_recommendations(user_id, model_name, int(num_results))
    if error:
        return f"<p style='color:red;'>⚠️ {error}</p>"

    model_colors = {'GloVe': '#27ae60', 'FastText': '#3498db', 'Word2Vec': '#e74c3c'}
    model_color = model_colors.get(model_name, '#667eea')
    hit_rate = model_hitrates.get(model_name, 0)
    n_liked = len(user_profile_df.iloc[user_id]['liked_song_indices'])

    result_html = f"""
    <div style="padding:15px; background:linear-gradient(135deg, {model_color} 0%, {model_color}dd 100%); 
                border-radius:10px; margin-bottom:20px;">
        <h2 style="color:white; margin:0;">🎯 Recommendations for User {user_id}</h2>
        <p style="color:#e0e0e0; margin:5px 0 0 0;">
            Model: <b>{model_name}</b> | Hit Rate: <b>{hit_rate}%</b> | 
            Based on {n_liked} favorite songs
        </p>
    </div>
    """

    for rank, song in enumerate(recommendations):
        if rank == 0:
            badge_color = "#FFD700"
            badge_text = "🥇 #1"
        elif rank == 1:
            badge_color = "#C0C0C0"
            badge_text = "🥈 #2"
        elif rank == 2:
            badge_color = "#CD7F32"
            badge_text = "🥉 #3"
        else:
            badge_color = model_color
            badge_text = f"#{rank+1}"

        result_html += f"""
        <div style="border:1px solid #e0e0e0; padding:15px; margin-bottom:15px; border-radius:12px; 
                    background:white; box-shadow:0 2px 8px rgba(0,0,0,0.1);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <span style="background:{badge_color}; color:white; padding:4px 12px; border-radius:20px; 
                             font-weight:bold; font-size:12px;">{badge_text}</span>
                <span style="background:#e8f5e9; color:#2e7d32; padding:4px 12px; border-radius:20px; 
                             font-size:12px;">Similarity: {song['similarity']:.4f}</span>
            </div>
            <div style="font-weight:bold; font-size:18px; color:#333; margin-bottom:5px;">
                🎵 {song['track_name']}
            </div>
            <div style="color:#666; font-size:14px; margin-bottom:10px;">
                👤 {song['artist_name']} &nbsp;|&nbsp; 🎸 {song['genre']}
            </div>
            <div style="background:#f5f5f5; padding:12px; border-radius:8px; 
                        border-left:4px solid {model_color};">
                <div style="font-size:11px; color:#888; margin-bottom:5px; font-weight:bold;">
                    📝 LYRICS:
                </div>
                <div style="font-style:italic; font-size:13px; color:#555; line-height:1.5;">
                    "{song['lyric_snippet']}"
                </div>
            </div>
        </div>
        """

    return result_html


def build_gradio_interface():
    """
    Membangun instance UI Gradio (Gradio Blocks).
    """
    available_models = list(model_data.keys())
    available_users = list(range(len(user_profile_df)))

    with gr.Blocks(
        title="🎵 Music Recommendation System",
        theme=gr.themes.Soft()
    ) as demo:

        gr.HTML("""
            <div style="text-align:center; padding:30px; background:linear-gradient(135deg, #506CE6 0%, #69389B 100%); 
                        border-radius:15px; margin-bottom:20px;">
                <h1 style="color:white; font-size:2.5em; margin:0;"><b>🎵 Music Recommendation System 🎵</b></h1>
                <p style="color:#e0e0e0; font-size:1.2em; margin:10px 0 0 0;">
                    Content-Based Filtering using GloVe Word Embedding
                </p>
                <p style="color:#b0b0b0; font-size:0.9em; margin:5px 0 0 0;">
                    Select Model → Select User → View History &amp; Recommendations
                </p>
            </div>
        """)

        with gr.Row():
            # Left column: Controls
            with gr.Column(scale=1):
                gr.HTML("<h3>⚙️ Configuration</h3>")

                model_input = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0] if available_models else "GloVe",
                    label="🤖 Select Model",
                    info="GloVe (Best), FastText, Word2Vec"
                )

                model_info_html = gr.HTML("""
                    <div style="background:#27ae6022; padding:10px; border-radius:8px; margin:10px 0; border-left:4px solid #27ae60;">
                        <b>🤖 GloVe</b> — Hit Rate: <b>92.65%</b>
                    </div>
                """)

                gr.HTML("<hr style='margin:15px 0;'>")
                gr.HTML("<h3>📝 Select User</h3>")

                user_id_input = gr.Dropdown(
                    choices=[str(i) for i in available_users],
                    value="0",
                    label="User ID",
                    info=f"Select user ID (0 – {len(available_users)-1})"
                )

                view_history_btn = gr.Button("👁️ View User History", variant="secondary", size="lg")

                gr.HTML("<br>")

                num_results = gr.Slider(
                    minimum=5,
                    maximum=20,
                    value=10,
                    step=1,
                    label="Number of Recommendations"
                )

                recommend_btn = gr.Button("🎯 Get Recommendations", variant="primary", size="lg")

                current_user_state = gr.State("")

                gr.HTML(f"""
                    <div style="background:#f5f5f5; padding:15px; border-radius:10px; margin-top:20px;">
                        <h4 style="margin:0 0 10px 0; color:#333;">📊 Model Comparison</h4>
                        <table style="width:100%; font-size:12px;">
                            <tr style="background:#46AD4F;">
                                <td style="padding:8px; color:white;"><b>🥇 GloVe</b></td>
                                <td style="padding:8px; text-align:right; color:white;"><b>92.65%</b></td>
                            </tr>
                            <tr style="background:#4584B1;">
                                <td style="padding:8px; color:white;">🥈 FastText</td>
                                <td style="padding:8px; text-align:right; color:white;">90.83%</td>
                            </tr>
                            <tr style="background:#A74C5A;">
                                <td style="padding:8px; color:white;">🥉 Word2Vec</td>
                                <td style="padding:8px; text-align:right; color:white;">90.44%</td>
                            </tr>
                        </table>
                    </div>
                    <div style="background:#e3f2fd; padding:15px; border-radius:10px; margin-top:15px; border-left:4px solid #1976d2;">
                        <h4 style="margin:0 0 10px 0; color:#1976d2;">ℹ️ How to Use</h4>
                        <ol style="margin:0; padding-left:20px; color:#1976d2; font-size:13px;">
                            <li>Select a <b>Model</b> from the dropdown</li>
                            <li>Select a <b>User ID</b></li>
                            <li>Click <b>"View User History"</b> to see liked songs &amp; lyrics</li>
                            <li>Click <b>"Get Recommendations"</b> to see suggestions</li>
                        </ol>
                    </div>
                """)

            # Right column: Output
            with gr.Column(scale=2):
                gr.HTML("<h3>📋 User History</h3>")
                history_output = gr.HTML(
                    value="<p style='color:#888; text-align:center; padding:30px;'>Select a User ID and click 'View User History'...</p>"
                )

                gr.HTML("<h3>🎶 Music Recommendations</h3>")
                recommend_output = gr.HTML(
                    value="<p style='color:#888; text-align:center; padding:30px;'>Recommendations will appear here after clicking 'Get Recommendations'...</p>"
                )

        # Event handlers
        def update_model_info(model_name):
            colors = {'GloVe': '#27ae60', 'FastText': '#3498db', 'Word2Vec': '#e74c3c'}
            hit_rate = model_hitrates.get(model_name, 0)
            color = colors.get(model_name, '#667eea')
            return f"""
                <div style="background:{color}22; padding:10px; border-radius:8px; margin:10px 0; border-left:4px solid {color};">
                    <b>🤖 {model_name}</b> — Hit Rate: <b>{hit_rate}%</b>
                </div>
            """

        model_input.change(
            fn=update_model_info,
            inputs=[model_input],
            outputs=[model_info_html]
        )

        view_history_btn.click(
            fn=display_user_info,
            inputs=[user_id_input],
            outputs=[history_output, current_user_state]
        )

        recommend_btn.click(
            fn=generate_recommendations_html,
            inputs=[user_id_input, model_input, num_results],
            outputs=recommend_output
        )

        gr.HTML("""
            <div style="text-align:center; padding:20px; margin-top:30px; border-top:1px solid #e0e0e0;">
                <p style="color:#888; font-size:12px;">
                    🎓 Music Recommendation System Based on User Listening History<br>
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
