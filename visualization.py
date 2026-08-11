"""
visualization.py — Fungsi-fungsi untuk visualisasi perbandingan model.

Modul ini berisi:
- plot_comparison_bar(): Bar chart perbandingan 3 metrik
- plot_radar_chart(): Radar chart perbandingan komprehensif
- plot_heatmap(): Heatmap semua metrik
- plot_per_fold(): Line chart per-fold performance
- plot_all_comparisons(): Jalankan semua visualisasi sekaligus
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import seaborn as sns

from config import VISUALISASI_PATH


# Palet warna konsisten
MODEL_COLORS = {
    'FastText': '#4f46e5',
    'GloVe': '#dc2626',
    'Word2Vec': '#059669'
}

# Style default
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 13,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafbfc',
})


def plot_comparison_bar(df_vis):
    """Bar chart perbandingan Hit Rate, MRR, NDCG."""
    colors = [MODEL_COLORS[m] for m in df_vis['Model']]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle('Perbandingan Metrik Evaluasi: FastText vs GloVe vs Word2Vec',
                 fontsize=18, fontweight='bold', y=1.02)

    metrics = [
        ('Hit Rate (%)', 'Hit Rate (%)', 'Persentase Hit Rate'),
        ('MRR', 'MRR', 'Mean Reciprocal Rank'),
        ('NDCG', 'NDCG', 'Normalized DCG'),
    ]

    for ax, (col, label, title) in zip(axes, metrics):
        bars = ax.bar(df_vis['Model'], df_vis[col], color=colors,
                      width=0.5, edgecolor='white', linewidth=1.5, zorder=3)
        for bar in bars:
            h = bar.get_height()
            fmt = f"{h:.2f}%" if '%' in label else f"{h:.4f}"
            ax.text(bar.get_x() + bar.get_width()/2., h + h*0.005,
                    fmt, ha='center', va='bottom', fontweight='bold', fontsize=11)

        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Model')
        ax.set_ylabel(label)
        ax.set_ylim([df_vis[col].min() * 0.97, df_vis[col].max() * 1.06])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=1)

    plt.tight_layout()
    path = os.path.join(VISUALISASI_PATH, 'comparison_bar_chart.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {path}")


def plot_radar_chart(df_vis):
    """Radar chart perbandingan komprehensif."""
    categories = ['Hit Rate', 'MRR', 'NDCG']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')

    for _, row in df_vis.iterrows():
        values = [row['Hit Rate'], row['MRR'], row['NDCG']]
        values += values[:1]
        color = MODEL_COLORS[row['Model']]
        ax.plot(angles, values, 'o-', linewidth=2.5, color=color,
                label=row['Model'], markersize=8)
        ax.fill(angles, values, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9, color='gray')
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_title('Radar Chart — Perbandingan Model\nFastText vs GloVe vs Word2Vec',
                 size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)

    plt.tight_layout()
    path = os.path.join(VISUALISASI_PATH, 'comparison_radar_chart.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {path}")


def plot_heatmap(df_vis):
    """Heatmap semua metrik evaluasi."""
    heat_data = df_vis.set_index('Model')[['Hit Rate', 'MRR', 'NDCG']]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('white')

    sns.heatmap(
        heat_data, annot=True, fmt='.4f', cmap='RdYlGn',
        linewidths=0.5, linecolor='white',
        ax=ax, annot_kws={'size': 13, 'weight': 'bold'},
        vmin=0.4, vmax=1.0
    )

    ax.set_title('Heatmap Perbandingan Metrik Evaluasi\n(FastText vs GloVe vs Word2Vec)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Metrik', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_xticklabels(['Hit Rate', 'MRR', 'NDCG'], fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

    plt.tight_layout()
    path = os.path.join(VISUALISASI_PATH, 'comparison_heatmap.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {path}")


def plot_per_fold(results, model_name):
    """Line chart per-fold performance untuk satu model."""
    fold_rows = [r for r in results['Fold Results'] if str(r['Fold']).startswith('Fold')]
    x = list(range(1, len(fold_rows) + 1))
    hr_vals = [float(r['Hit Rate (%)']) / 100 for r in fold_rows]
    mrr_vals = [float(r['MRR']) for r in fold_rows]
    ndcg_vals = [float(r['NDCG']) for r in fold_rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8fafc')
    ax.plot(x, hr_vals,   'o-', color='#4f46e5', linewidth=2.5, markersize=9, label='Hit Rate')
    ax.plot(x, mrr_vals,  's-', color='#f43f5e', linewidth=2.5, markersize=9, label='MRR')
    ax.plot(x, ndcg_vals, '^-', color='#10b981', linewidth=2.5, markersize=9, label='NDCG')

    ax.set_title(f'{model_name} — Per-Fold Performance', fontsize=15, fontweight='bold')
    ax.set_xlabel('Fold', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([r['Fold'] for r in fold_rows])
    ax.set_ylim([0, 1.18])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.4, linestyle='--')

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    path = os.path.join(VISUALISASI_PATH, f'{model_name.lower()}_per_fold.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"✅ Saved: {path}")


def plot_per_fold_comparison(all_results):
    """
    Line chart per-fold yang membandingkan SEMUA model dalam 1 figure.
    Format sama persis dengan recommendersysMusic.ipynb:
    - 3 subplot: Hit Rate (normalized), MRR, NDCG
    - Hit Rate dinormalisasi ke skala 0-1
    - Annotasi: persentase untuk Hit Rate, 4 desimal untuk MRR/NDCG
    """
    # Kumpulkan data per-fold dari setiap model
    model_fold_data = {}
    for name, data in all_results.items():
        fold_rows = [r for r in data['results']['Fold Results']
                     if str(r['Fold']).startswith('Fold')]
        model_fold_data[name] = {
            'folds': [r['Fold'] for r in fold_rows],
            'hit_rate_pct': [float(r['Hit Rate (%)']) for r in fold_rows],
            'hit_rate_norm': [float(r['Hit Rate (%)']) / 100.0 for r in fold_rows],
            'mrr': [float(r['MRR']) for r in fold_rows],
            'ndcg': [float(r['NDCG']) for r in fold_rows],
        }

    n_folds = len(list(model_fold_data.values())[0]['folds'])
    x = list(range(1, n_folds + 1))
    fold_labels = list(model_fold_data.values())[0]['folds']

    # Style per model — sama persis dengan referensi
    line_styles = {
        'FastText': {'color': '#3333CC', 'marker': 'o', 'linestyle': '-'},
        'GloVe':    {'color': '#CC0000', 'marker': 's', 'linestyle': '-'},
        'Word2Vec': {'color': '#009933', 'marker': '^', 'linestyle': '-'},
    }

    # 3 metrik: (data_key, y_label, annotation_format)
    metrics = [
        ('hit_rate_norm', 'Hit Rate (normalized)', lambda v, d: f"{d:.2f}%"),
        ('mrr',           'MRR',                   lambda v, d: f"{v:.4f}"),
        ('ndcg',          'NDCG',                  lambda v, d: f"{v:.4f}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(23, 8))
    fig.patch.set_facecolor('white')

    # Ukuran font diperbesar & ditebalkan
    DATA_LABEL_FONT_SIZE = 15
    AXIS_TITLE_FONT_SIZE = 18
    TICK_LABEL_FONT_SIZE = 16
    LEGEND_FONT_SIZE = 16

    for ax, (key, ylabel, fmt_fn) in zip(axes, metrics):
        ax.set_facecolor('white')

        all_vals = []
        for name in all_results:
            style = line_styles.get(name, {'color': 'gray', 'marker': 'o', 'linestyle': '-'})
            vals = model_fold_data[name][key]
            all_vals.extend(vals)

            ax.plot(x, vals, marker=style['marker'], linestyle=style['linestyle'],
                    color=style['color'], linewidth=3.2, markersize=11,
                    label=name, zorder=3)

        # Efek stroke putih di sekeliling teks agar angka tidak bertabrakan dengan garis kurva
        stroke_effect = [path_effects.withStroke(linewidth=4.5, foreground='white')]

        # Annotasi di setiap titik data (posisi dinamis berdasarkan urutan nilai & gap)
        for i_idx, xi in enumerate(x):
            fold_values = []
            for name in all_results:
                v = model_fold_data[name][key][i_idx]
                p = model_fold_data[name]['hit_rate_pct'][i_idx]
                fold_values.append((v, p, name))
            
            # Urutkan model berdasarkan nilai (descending) pada fold xi
            fold_values.sort(key=lambda item: item[0], reverse=True)

            v_top, _, _ = fold_values[0]
            v_mid, _, _ = fold_values[1]
            v_bot, _, _ = fold_values[2]

            gap_top = v_top - v_mid
            gap_bot = v_mid - v_bot

            for rank, (vi, pi, name) in enumerate(fold_values):
                style = line_styles.get(name, {'color': 'gray'})
                label_text = fmt_fn(vi, pi)

                if rank == 0:
                    xytext = (0, 15)
                    va = 'bottom'
                elif rank == 2:
                    xytext = (0, -20)
                    va = 'top'
                else:
                    if gap_top >= gap_bot:
                        xytext = (0, 15)
                        va = 'bottom'
                    else:
                        xytext = (0, -20)
                        va = 'top'

                ax.annotate(label_text, (xi, vi),
                            textcoords='offset points', xytext=xytext,
                            ha='center', va=va, fontsize=DATA_LABEL_FONT_SIZE, fontweight='bold',
                            color=style['color'], path_effects=stroke_effect, zorder=5)

        # Y-axis: margin ketat namun cukup (38%) untuk memberi ruang teks atas dan bawah
        y_min = min(all_vals)
        y_max = max(all_vals)
        y_range = y_max - y_min
        ax.set_ylim([y_min - y_range * 0.38, y_max + y_range * 0.38])

        ax.set_xlabel('Fold', fontsize=AXIS_TITLE_FONT_SIZE, fontweight='bold', labelpad=12)
        ax.set_ylabel(ylabel, fontsize=AXIS_TITLE_FONT_SIZE, fontweight='bold', labelpad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(fold_labels, fontsize=TICK_LABEL_FONT_SIZE, fontweight='bold')
        ax.tick_params(axis='y', labelsize=TICK_LABEL_FONT_SIZE)
        
        for tick_lbl in ax.get_yticklabels():
            tick_lbl.set_fontweight('bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.8)
        ax.spines['bottom'].set_linewidth(1.8)
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=1.0, zorder=1)
        
        leg = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                        ncol=3, fontsize=LEGEND_FONT_SIZE, frameon=False)
        for text in leg.get_texts():
            text.set_fontweight('bold')

    plt.tight_layout()
    path = os.path.join(VISUALISASI_PATH, 'per_fold_comparison_all_models.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"✅ Saved: {path}")


def plot_all_comparisons(all_results):
    """
    Jalankan semua visualisasi sekaligus.
    
    Args:
        all_results: dict {model_name: {results: {...}, embeddings: ...}}
    """
    # Bangun DataFrame
    rows = []
    for name, data in all_results.items():
        r = data['results']
        rows.append({
            'Model': name,
            'Hit Rate': r['Hit Rate'],
            'MRR': r['MRR'],
            'NDCG': r['NDCG'],
            'Hit Rate (%)': r['Hit Rate'] * 100,
        })
    df_vis = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("📊 VISUALISASI PERBANDINGAN MODEL")
    print("=" * 60)

    plot_comparison_bar(df_vis)
    plot_radar_chart(df_vis)
    plot_heatmap(df_vis)

    # Per-fold comparison (semua model dalam 1 figure)
    plot_per_fold_comparison(all_results)

    # Per-fold untuk setiap model (individual)
    for name, data in all_results.items():
        plot_per_fold(data['results'], name)

    print("\n✅ Semua visualisasi selesai dibuat!")
