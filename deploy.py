"""
deploy.py — Script untuk otomatisasi konversi dan deployment Gradio UI ke Hugging Face Space.
"""

import os
import subprocess
import pandas as pd

def deploy_to_huggingface(commit_message="Update app.py from main.ipynb"):
    import sys
    # Reconfigure stdout to utf-8 to prevent UnicodeEncodeError on Windows terminals with emojis
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
        try:
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    """
    Membaca gradio_app.py lokal, mengonversinya menjadi app.py versi Hugging Face,
    kemudian melakukan commit & push ke Hugging Face Space secara otomatis.
    """
    # Path file sumber dan tujuan
    GRADIO_APP_SOURCE = 'gradio_app.py'
    
    # Dapatkan path folder deploy di Google Drive
    HF_DEPLOY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'huggingface_deploy')
    if not os.path.exists(HF_DEPLOY_PATH):
        HF_DEPLOY_PATH = r'G:\My Drive\FinalArc\research\huggingface_deploy'

    app_py_target = os.path.join(HF_DEPLOY_PATH, 'app.py')

    if not os.path.exists(GRADIO_APP_SOURCE):
        # Coba cari dengan absolute path jika running di context berbeda
        GRADIO_APP_SOURCE = r'G:\My Drive\FinalArc\research\gradio_app.py'

    if not os.path.exists(GRADIO_APP_SOURCE):
        print(f"❌ File sumber {GRADIO_APP_SOURCE} tidak ditemukan!")
        return

    print(f"📖 Membaca sumber: {GRADIO_APP_SOURCE}")
    with open(GRADIO_APP_SOURCE, 'r', encoding='utf-8') as f:
        source_code = f.read()

    # 1. Ganti import config lokal dengan path konstan Hugging Face
    converted_code = source_code.replace(
        "from config import EMBEDDINGS_PATH",
        'EMBEDDINGS_PATH = "./embeddings"\nimport ast'
    )

    # 2. Tambahkan block eksekusi standalone di bagian bawah
    hf_execution_block = """

# ==============================================================================
# STANDALONE EXECUTION FOR HUGGING FACE SPACE
# ==============================================================================
if __name__ == "__main__":
    print("📂 Loading datasets for HuggingFace Space...")
    
    # Load metadata lagu
    merged_songs_path = "./data/merged_songs.csv"
    if os.path.exists(merged_songs_path):
        merged_df = pd.read_csv(merged_songs_path)
        setup_gradio_data(merged_df)
    else:
        print(f"❌ File {merged_songs_path} tidak ditemukan!")
        
    # Load user profiles dari CSV (bukan generate acak)
    user_profiles_path = "./data/user_profiles.csv"
    if os.path.exists(user_profiles_path):
        user_profile_df = pd.read_csv(user_profiles_path)
        user_profile_df['liked_song_indices'] = user_profile_df['liked_song_indices'].apply(ast.literal_eval)
        user_profile_df['play_counts'] = user_profile_df['play_counts'].apply(ast.literal_eval)
        print(f"✅ Loaded {len(user_profile_df)} user profiles from CSV")
    else:
        print(f"❌ File {user_profiles_path} tidak ditemukan!")

    # Build dan Launch UI
    demo = build_gradio_interface()
    demo.launch()
"""

    converted_code += hf_execution_block

    # Tulis ke app.py
    os.makedirs(os.path.dirname(app_py_target), exist_ok=True)
    with open(app_py_target, 'w', encoding='utf-8') as f:
        f.write(converted_code)

    print(f"✅ app.py berhasil dikonversi dan ditulis ke: {app_py_target}")

    # 3. Proses Git Push ke HuggingFace Space
    try:
        print("\n🚀 Memulai proses deployment ke Hugging Face Space...")
        
        # Git status
        print("🔍 1. Memeriksa status git...")
        subprocess.run(["git", "status", "-s"], cwd=HF_DEPLOY_PATH, check=True)
        
        # Git add
        print("➕ 2. Menambahkan perubahan (git add app.py)...")
        subprocess.run(["git", "add", "app.py"], cwd=HF_DEPLOY_PATH, check=True)
        
        # Git commit
        print(f"💾 3. Melakukan commit: '{commit_message}'...")
        subprocess.run(["git", "config", "user.name", "Faiq Misbah Yazdi"], cwd=HF_DEPLOY_PATH)
        subprocess.run(["git", "config", "user.email", "faiq.misbah.y@gmail.com"], cwd=HF_DEPLOY_PATH)
        
        commit_res = subprocess.run(["git", "commit", "-m", commit_message], cwd=HF_DEPLOY_PATH, capture_output=True, text=True)
        if "nothing to commit" in commit_res.stdout or "nothing to commit" in commit_res.stderr:
            print("   ℹ️ Tidak ada perubahan baru di app.py yang perlu di-commit.")
        else:
            print(commit_res.stdout)
            
        # Git push
        print("📤 4. Melakukan push ke Hugging Face (git push)...")
        push_res = subprocess.run(["git", "push", "origin", "main"], cwd=HF_DEPLOY_PATH, capture_output=True, text=True)
        
        if push_res.returncode == 0:
            print("\n🎉 BERHASIL! Perubahan berhasil di-push ke Hugging Face Space.")
            print("🔗 Silakan cek: https://huggingface.co/spaces/faiqmisbah/musik-rekomendasi")
        else:
            print("\n❌ Gagal melakukan push. Error:")
            print(push_res.stderr)
            print(push_res.stdout)
            
    except Exception as e:
        print(f"❌ Terjadi error saat deployment: {e}")
