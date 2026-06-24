"""
analisi_report.py
=================
Script unico per generare tutti i dati e le figure necessari al report.

Cosa produce:
  results/01_broadband_proxy_error.csv   — motivazione stima per-banda
  results/02_mape_per_bracket.csv        — MAPE e MAE per bracket T30
  results/03_mae_per_room.csv            — MAE per tipo di stanza
  results/04_vad_subset_analysis.csv     — errore su campioni T30 > 3.5s
  results/05_training_history.csv        — history del retraining (con ablation)
  results/fig_broadband_proxy.png        — figura per il report
  results/fig_mape_bracket.png           — figura per il report
  results/fig_mae_room.png               — figura per il report
  results/fig_learning_curves.png        — figure per il report
  modello_symmetric_ablation.h5          — modello kernel simmetrici (per confronto)

Esegui dalla cartella radice del progetto:
  python scripts/analisi_report.py

Richiede: numpy, pandas, matplotlib, tensorflow, scikit-learn
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # headless, nessuna finestra

# Crea la cartella risultati
os.makedirs("results", exist_ok=True)

CSV_CLEAN = "dataset_ml_clean.csv"
DIR_SPEC  = "dataset_spectrograms"
MODEL_ORIGINAL = "modello_xr_t30.h5"
MODEL_SYMMETRIC = "modello_symmetric_ablation.h5"
TARGET_FRAMES = 512

print("=" * 60)
print("FASE 1 — Analisi dal CSV (nessuna GPU necessaria)")
print("=" * 60)

df = pd.read_csv(CSV_CLEAN)
print(f"Dataset: {len(df)} campioni, {df['room_type'].nunique()} tipologie\n")

# ─────────────────────────────────────────────────────────────
# 1. BROADBAND PROXY ERROR — motivazione stima per-banda
# ─────────────────────────────────────────────────────────────
print("1. Calcolo errore broadband-as-proxy per ogni banda...")

results_proxy = {}
for col in ['t30_1000Hz', 't30_2000Hz', 't30_4000Hz']:
    diff = (df['t30_broadband'] - df[col]).abs()
    results_proxy[col] = {
        'MAE_proxy':   round(diff.mean(), 3),
        'max_error':   round(diff.max(), 3),
        'p95_error':   round(diff.quantile(0.95), 3),
        'n_over_1s':   int((diff > 1.0).sum()),
        'pct_over_1s': round((diff > 1.0).mean() * 100, 1)
    }

df_proxy = pd.DataFrame(results_proxy).T
df_proxy.index.name = 'band'
df_proxy.to_csv("results/01_broadband_proxy_error.csv")
print(df_proxy.to_string())
print()

# Figura
fig, ax = plt.subplots(figsize=(7, 4))
bands = ['1 kHz', '2 kHz', '4 kHz']
mae_vals = [results_proxy[k]['MAE_proxy'] for k in ['t30_1000Hz','t30_2000Hz','t30_4000Hz']]
colors = ['#d62728', '#ff7f0e', '#2ca02c']
bars = ax.bar(bands, mae_vals, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
ax.axhline(y=0.455, color='navy', linestyle='--', linewidth=1.2, label='CNN MAE @ 1kHz')
ax.axhline(y=0.362, color='steelblue', linestyle='--', linewidth=1.2, label='CNN MAE @ 2kHz')
ax.axhline(y=0.320, color='teal', linestyle='--', linewidth=1.2, label='CNN MAE @ 4kHz')
for bar, val in zip(bars, mae_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('MAE (s)', fontsize=11)
ax.set_title('Broadband as per-band proxy vs CNN per-band estimation', fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, max(mae_vals) * 1.25)
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig("results/fig_broadband_proxy.png", dpi=150, bbox_inches='tight')
plt.close()
print("  → results/fig_broadband_proxy.png\n")

# ─────────────────────────────────────────────────────────────
# 2. MAPE + MAE PER BRACKET T30 @ 1kHz
# ─────────────────────────────────────────────────────────────
print("2. MAPE e MAE per bracket T30...")

# I valori MAE per banda sono quelli documentati dai tuoi esperimenti
# Usiamo il ground truth del CSV + MAE documentato per stimare MAPE
# Nota: senza le predizioni salvate, calcoliamo il MAPE "atteso" dalla distribuzione
# dei dati nel test set (random_state=42, 20% split)

from sklearn.model_selection import train_test_split
_, df_test = train_test_split(df, test_size=0.2, random_state=42)
print(f"  Test set: {len(df_test)} campioni (split identico al training)\n")

bands_info = {
    't30_1000Hz': {'mae': 0.455, 'label': '1 kHz'},
    't30_2000Hz': {'mae': 0.362, 'label': '2 kHz'},
    't30_4000Hz': {'mae': 0.320, 'label': '4 kHz'},
}

bins   = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0, 12.0]
labels = ['0–0.5', '0.5–1', '1–1.5', '1.5–2', '2–2.5', '2.5–3', '3–3.5', '3.5–5', '>5']

rows = []
for col, info in bands_info.items():
    df_test_copy = df_test.copy()
    df_test_copy['bracket'] = pd.cut(df_test_copy[col], bins=bins, labels=labels, right=False)
    gt = df_test_copy[col]
    # Stima MAPE per bracket usando l'errore relativo medio del ground truth
    # (proxy conservativo: assumiamo che l'errore assoluto sia ~uniformemente distribuito = MAE globale)
    for bracket in labels:
        sub = df_test_copy[df_test_copy['bracket'] == bracket]
        if len(sub) == 0:
            continue
        gt_sub = sub[col]
        mean_gt = gt_sub.mean()
        # MAPE stimato = MAE_globale / mean_T30_bracket * 100
        mape_est = (info['mae'] / mean_gt * 100) if mean_gt > 0.01 else float('nan')
        rows.append({
            'band':       info['label'],
            'bracket':    bracket,
            'n_samples':  len(sub),
            'mean_T30':   round(mean_gt, 3),
            'MAPE_est_%': round(mape_est, 1),
            'MAE_global': info['mae'],
        })

df_bracket = pd.DataFrame(rows)
df_bracket.to_csv("results/02_mape_per_bracket.csv", index=False)

# Stampa tabella sintetica per 1kHz
print("  MAPE stimato per bracket (1 kHz):")
sub1k = df_bracket[df_bracket['band'] == '1 kHz'][['bracket','n_samples','mean_T30','MAPE_est_%']]
print(sub1k.to_string(index=False))
print()

# Figura: MAPE per bracket, tre bande
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(labels))
width = 0.28
colors_bands = {'1 kHz': '#d62728', '2 kHz': '#ff7f0e', '4 kHz': '#2ca02c'}
for i, (band_label, color) in enumerate(colors_bands.items()):
    sub = df_bracket[df_bracket['band'] == band_label].set_index('bracket').reindex(labels)
    ax.bar(x + (i-1)*width, sub['MAPE_est_%'].fillna(0),
           width=width, label=band_label, color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
ax.axhline(y=5, color='black', linestyle=':', linewidth=1, label='JND threshold (~5%)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Estimated MAPE (%)', fontsize=11)
ax.set_title('Estimated MAPE per T30 bracket (based on global MAE)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig("results/fig_mape_bracket.png", dpi=150, bbox_inches='tight')
plt.close()
print("  → results/fig_mape_bracket.png\n")

# ─────────────────────────────────────────────────────────────
# 3. MAE PER TIPO DI STANZA (stima da distribuzione test set)
# ─────────────────────────────────────────────────────────────
print("3. Distribuzione campioni per stanza nel test set...")

room_counts = df_test['room_type'].value_counts().sort_index()
room_t30_stats = df_test.groupby('room_type')[['t30_1000Hz','t30_2000Hz','t30_4000Hz']].agg(['mean','std'])
room_t30_stats.columns = ['_'.join(c) for c in room_t30_stats.columns]
room_t30_stats['n_test'] = room_counts
room_t30_stats.to_csv("results/03_mae_per_room.csv")
print(room_t30_stats[['n_test','t30_1000Hz_mean','t30_1000Hz_std']].to_string())
print()
print("  NOTA: per il MAE effettivo per stanza è necessario caricare il modello.")
print("  Se vuoi saltare questo (richiede TensorFlow), commenta la FASE 2 sotto.\n")

# Figura distribuzione campioni per stanza
fig, ax = plt.subplots(figsize=(8, 4))
rooms = room_counts.index.tolist()
counts = room_counts.values
colors_rooms = plt.cm.tab10(np.linspace(0, 1, len(rooms)))
ax.barh(rooms, counts, color=colors_rooms, edgecolor='black', linewidth=0.6)
for i, (room, count) in enumerate(zip(rooms, counts)):
    ax.text(count + 0.3, i, str(count), va='center', fontsize=9)
ax.set_xlabel('Number of test samples', fontsize=11)
ax.set_title('Test set distribution by room type', fontsize=11)
ax.grid(axis='x', alpha=0.4)
plt.tight_layout()
plt.savefig("results/fig_mae_room.png", dpi=150, bbox_inches='tight')
plt.close()
print("  → results/fig_mae_room.png\n")

# ─────────────────────────────────────────────────────────────
# 4. VAD SUBSET ANALYSIS — campioni T30 > 3.5s
# ─────────────────────────────────────────────────────────────
print("4. Analisi subset T30 > 3.5s (limite VAD)...")

high_reverb = df_test[df_test['t30_1000Hz'] > 3.5]
normal      = df_test[df_test['t30_1000Hz'] <= 3.5]

vad_results = pd.DataFrame({
    'subset': ['T30 <= 3.5s (VAD valid)', 'T30 > 3.5s (VAD limit)'],
    'n_samples': [len(normal), len(high_reverb)],
    'mean_T30_1kHz': [round(normal['t30_1000Hz'].mean(), 3),
                      round(high_reverb['t30_1000Hz'].mean(), 3)],
    'mean_T30_4kHz': [round(normal['t30_4000Hz'].mean(), 3),
                      round(high_reverb['t30_4000Hz'].mean(), 3)],
    'pct_of_test':   [round(len(normal)/len(df_test)*100, 1),
                      round(len(high_reverb)/len(df_test)*100, 1)],
    'room_types':    [
        ', '.join(normal['room_type'].unique()),
        ', '.join(high_reverb['room_type'].unique())
    ]
})
vad_results.to_csv("results/04_vad_subset_analysis.csv", index=False)
print(vad_results[['subset','n_samples','mean_T30_1kHz','pct_of_test']].to_string(index=False))
print()

# ─────────────────────────────────────────────────────────────
# FASE 2 — Retraining + Ablation + History
# (richiede TensorFlow e dataset_spectrograms/)
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("FASE 2 — Retraining con salvataggio history + ablation kernel")
print("=" * 60)

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    print(f"TensorFlow {tf.__version__} trovato.\n")
except ImportError:
    print("TensorFlow non trovato. Installa con: pip install tensorflow")
    print("Fase 1 completata. Esci.")
    sys.exit(0)

# ── Carica dati ──────────────────────────────────────────────
def load_spectrograms(df, dir_spec, target_frames=512):
    X, y, valid_idx = [], [], []
    missing = 0
    for idx, row in df.iterrows():
        spec_filename = row['wet_audio_file'].replace('.wav', '.npy')
        spec_path = os.path.join(dir_spec, spec_filename)
        if not os.path.exists(spec_path):
            missing += 1
            continue
        spec = np.load(spec_path)
        if spec.shape[1] < target_frames:
            spec = np.pad(spec, ((0,0),(0, target_frames - spec.shape[1])),
                         mode='constant', constant_values=-5.0)
        else:
            spec = spec[:, :target_frames]
        X.append(spec)
        y.append([row['t30_1000Hz'], row['t30_2000Hz'], row['t30_4000Hz']])
        valid_idx.append(idx)
    if missing > 0:
        print(f"  ATTENZIONE: {missing} spettrogrammi mancanti, ignorati.")
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    return X, y, valid_idx

print("Caricamento spettrogrammi...")
X, y, valid_idx = load_spectrograms(df, DIR_SPEC, TARGET_FRAMES)
print(f"X shape: {X.shape}, y shape: {y.shape}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
input_shape = (X.shape[1], X.shape[2], 1)

# ── Funzione build model ─────────────────────────────────────
def build_asymmetric(input_shape):
    """Architettura originale con kernel asimmetrici (3,7)-(3,5)-(3,3)."""
    m = models.Sequential([
        layers.Conv2D(32, (3,7), input_shape=input_shape, padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D((2,3)),
        layers.Conv2D(64, (3,5), padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128), layers.BatchNormalization(),
        layers.Activation('relu'), layers.Dropout(0.4),
        layers.Dense(3, activation='linear'),
    ])
    m.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=['mae'])
    return m

def build_symmetric(input_shape):
    """Architettura ablation: tutti i kernel (3,3)."""
    m = models.Sequential([
        layers.Conv2D(32, (3,3), input_shape=input_shape, padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D((2,3)),
        layers.Conv2D(64, (3,3), padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128), layers.BatchNormalization(),
        layers.Activation('relu'), layers.Dropout(0.4),
        layers.Dense(3, activation='linear'),
    ])
    m.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=['mae'])
    return m

cb_list = lambda name: [
    callbacks.ModelCheckpoint(name, save_best_only=True, monitor='val_mae', mode='min'),
    callbacks.EarlyStopping(monitor='val_mae', patience=15, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=5,
                                min_lr=1e-6, verbose=0),
]

# ── Train ASYMMETRIC (recupero history) ──────────────────────
print("Training ASIMMETRICO (architettura originale)...")
print("Questo richiede gli stessi minuti del training originale.\n")

model_asym = build_asymmetric(input_shape)
history_asym = model_asym.fit(
    X_train, y_train,
    epochs=80, batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=cb_list("modello_retraining_asym.h5"),
    verbose=1
)

# Carica il best checkpoint
model_asym.load_weights("modello_retraining_asym.h5")
_, mae_asym_1k = model_asym.evaluate(X_test, y_test, verbose=0)

# MAE per banda
preds_asym = model_asym.predict(X_test, verbose=0)
mae_asym_bands = np.mean(np.abs(preds_asym - y_test), axis=0)
print(f"\n✅ ASIMMETRICO — MAE: 1kHz={mae_asym_bands[0]:.3f}s | "
      f"2kHz={mae_asym_bands[1]:.3f}s | 4kHz={mae_asym_bands[2]:.3f}s\n")

# ── Train SYMMETRIC (ablation) ────────────────────────────────
print("Training SIMMETRICO (ablation — kernel 3x3)...\n")

model_sym = build_symmetric(input_shape)
history_sym = model_sym.fit(
    X_train, y_train,
    epochs=80, batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=cb_list(MODEL_SYMMETRIC),
    verbose=1
)

model_sym.load_weights(MODEL_SYMMETRIC)
preds_sym = model_sym.predict(X_test, verbose=0)
mae_sym_bands = np.mean(np.abs(preds_sym - y_test), axis=0)
print(f"\n✅ SIMMETRICO  — MAE: 1kHz={mae_sym_bands[0]:.3f}s | "
      f"2kHz={mae_sym_bands[1]:.3f}s | 4kHz={mae_sym_bands[2]:.3f}s\n")

# ── MAE per stanza dal modello asimmetrico ────────────────────
print("Calcolo MAE per tipo di stanza (modello asimmetrico)...")
_, _, valid_idx_arr = load_spectrograms(df, DIR_SPEC, TARGET_FRAMES)
df_valid = df.loc[valid_idx_arr].copy()
_, y_test_full, valid_idx_test = load_spectrograms(df.loc[valid_idx_arr], DIR_SPEC, TARGET_FRAMES)

# Ricostruiamo il test set con gli indici originali
X_all, y_all, _ = load_spectrograms(df, DIR_SPEC, TARGET_FRAMES)
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
df_tr, df_te = train_test_split(df_valid, test_size=0.2, random_state=42)

preds_te = model_asym.predict(X_te, verbose=0)
df_te = df_te.copy()
df_te['pred_1k'] = preds_te[:,0]
df_te['pred_2k'] = preds_te[:,1]
df_te['pred_4k'] = preds_te[:,2]
df_te['ae_1k'] = np.abs(df_te['pred_1k'] - df_te['t30_1000Hz'])
df_te['ae_2k'] = np.abs(df_te['pred_2k'] - df_te['t30_2000Hz'])
df_te['ae_4k'] = np.abs(df_te['pred_4k'] - df_te['t30_4000Hz'])

mae_room = df_te.groupby('room_type')[['ae_1k','ae_2k','ae_4k']].mean().round(3)
mae_room.columns = ['MAE_1kHz','MAE_2kHz','MAE_4kHz']
mae_room['n_test'] = df_te.groupby('room_type').size()
mae_room.to_csv("results/03_mae_per_room.csv")
print(mae_room.to_string())
print()

# ── Salva history completa ────────────────────────────────────
hist_asym_df = pd.DataFrame(history_asym.history)
hist_asym_df['model']  = 'asymmetric'
hist_asym_df['epoch']  = hist_asym_df.index + 1
hist_sym_df  = pd.DataFrame(history_sym.history)
hist_sym_df['model']   = 'symmetric_ablation'
hist_sym_df['epoch']   = hist_sym_df.index + 1
hist_all = pd.concat([hist_asym_df, hist_sym_df], ignore_index=True)
hist_all.to_csv("results/05_training_history.csv", index=False)
print("  → results/05_training_history.csv\n")

# ── Ablation comparison table ─────────────────────────────────
ablation_table = pd.DataFrame({
    'model':       ['Asymmetric (3×7, 3×5, 3×3)', 'Symmetric (3×3, 3×3, 3×3)'],
    'MAE_1kHz':    [round(mae_asym_bands[0], 3), round(mae_sym_bands[0], 3)],
    'MAE_2kHz':    [round(mae_asym_bands[1], 3), round(mae_sym_bands[1], 3)],
    'MAE_4kHz':    [round(mae_asym_bands[2], 3), round(mae_sym_bands[2], 3)],
    'MAE_avg':     [round(mae_asym_bands.mean(), 3), round(mae_sym_bands.mean(), 3)],
})
ablation_table.to_csv("results/06_ablation_comparison.csv", index=False)
print("ABLATION STUDY RESULTS:")
print(ablation_table.to_string(index=False))
print()

# ── Figure learning curves ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Val MAE curve
axes[0].plot(hist_asym_df['epoch'], hist_asym_df['val_mae'],
             label='Asymmetric (3×7)', color='#1f77b4', linewidth=2)
axes[0].plot(hist_sym_df['epoch'],  hist_sym_df['val_mae'],
             label='Symmetric (3×3)', color='#d62728', linewidth=2, linestyle='--')
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Validation MAE (s)', fontsize=11)
axes[0].set_title('Validation MAE — asymmetric vs symmetric kernels', fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.4)

# Training loss curve (asimmetrico)
axes[1].plot(hist_asym_df['epoch'], hist_asym_df['loss'],
             label='Train loss', color='#1f77b4', linewidth=2)
axes[1].plot(hist_asym_df['epoch'], hist_asym_df['val_loss'],
             label='Val loss', color='#1f77b4', linewidth=2, linestyle='--', alpha=0.7)
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Huber Loss', fontsize=11)
axes[1].set_title('Training vs validation loss (asymmetric model)', fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.4)

plt.tight_layout()
plt.savefig("results/fig_learning_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("  → results/fig_learning_curves.png")

# ── Figura MAE per stanza ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(mae_room))
w = 0.28
ax.bar(x - w, mae_room['MAE_1kHz'], w, label='1 kHz', color='#d62728', alpha=0.85,
       edgecolor='black', linewidth=0.6)
ax.bar(x,     mae_room['MAE_2kHz'], w, label='2 kHz', color='#ff7f0e', alpha=0.85,
       edgecolor='black', linewidth=0.6)
ax.bar(x + w, mae_room['MAE_4kHz'], w, label='4 kHz', color='#2ca02c', alpha=0.85,
       edgecolor='black', linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(mae_room.index, rotation=25, ha='right', fontsize=10)
ax.set_ylabel('MAE (s)', fontsize=11)
ax.set_title('Per-band MAE by room type (asymmetric model)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig("results/fig_mae_room.png", dpi=150, bbox_inches='tight')
plt.close()
print("  → results/fig_mae_room.png (aggiornata con valori reali)\n")

print("=" * 60)
print("COMPLETATO. Tutti i file sono in results/")
print("=" * 60)
print()
print("File pronti per il report:")
print("  results/01_broadband_proxy_error.csv")
print("  results/02_mape_per_bracket.csv")
print("  results/03_mae_per_room.csv")
print("  results/04_vad_subset_analysis.csv")
print("  results/05_training_history.csv")
print("  results/06_ablation_comparison.csv")
print("  results/fig_broadband_proxy.png")
print("  results/fig_mape_bracket.png")
print("  results/fig_mae_room.png")
print("  results/fig_learning_curves.png")
