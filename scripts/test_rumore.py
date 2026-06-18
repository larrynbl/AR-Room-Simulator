import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- CONFIGURAZIONE ---
CSV_CLEAN = "dataset_ml_clean.csv"
DIR_WET = "dataset_wet"
MODEL_NAME = "modello_xr_t30.h5"
TARGET_FRAMES = 512

# Livelli di rumore da testare (in dB SNR). Più è basso, più c'è rumore!
# 'Clean' significa il file originale senza rumore aggiunto.
SNR_LEVELS = ['Clean', 30, 20, 10]

def genera_rumore_rosa(lunghezza):
    """Genera rumore rosa (1/f) usando la trasformata di Fourier (FFT)"""
    uneven = lunghezza % 2
    X = np.random.randn(lunghezza // 2 + 1 + uneven) + 1j * np.random.randn(lunghezza // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1.) # Filtro 1/f
    y = (np.fft.irfft(X / S)).real
    if uneven:
        y = y[:-1]
    return y

def mixa_rumore(segnale, snr_db):
    """Aggiunge rumore rosa al segnale rispettando un preciso SNR"""
    rumore = genera_rumore_rosa(len(segnale))
    
    # Calcolo dell'energia (Root Mean Square)
    rms_segnale = np.sqrt(np.mean(segnale**2))
    rms_rumore = np.sqrt(np.mean(rumore**2))
    
    # Calcoliamo quanto deve essere forte il rumore per ottenere l'SNR voluto
    rms_rumore_target = rms_segnale / (10 ** (snr_db / 20))
    rumore_scalato = rumore * (rms_rumore_target / (rms_rumore + 1e-8))
    
    return segnale + rumore_scalato

def estrai_feature_al_volo(y, sr=16000):
    """Riproduce ESATTAMENTE la nostra pipeline di estrazione con Z-Score"""
    intervals = librosa.effects.split(y, top_db=40)
    if len(intervals) > 0:
        y_aligned = y[intervals[0][0]:]
    else:
        y_aligned = y

    mel_spec = librosa.feature.melspectrogram(y=y_aligned, sr=sr, n_fft=2048, hop_length=256, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)

    # Z-Score
    mean_val = np.mean(mel_spec_db)
    std_val = np.std(mel_spec_db)
    norm_spec = (mel_spec_db - mean_val) / (std_val + 1e-8)

    # Padding
    if norm_spec.shape[1] < TARGET_FRAMES:
        pad_width = TARGET_FRAMES - norm_spec.shape[1]
        norm_spec = np.pad(norm_spec, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=-5.0)
    else:
        norm_spec = norm_spec[:, :TARGET_FRAMES]
        
    return norm_spec

def main():
    print("🚀 Inizio Test di Robustezza al Rumore Rosa (Pink Noise)")
    
    # 1. Carichiamo solo la lista dei file per fare lo split (non carichiamo gli .npy)
    df = pd.read_csv(CSV_CLEAN)
    # Stesso random_state=42 dell'addestramento per essere sicuri di usare il Test Set
    _, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"🎓 File da elaborare per il test alla cieca: {len(df_test)}")
    
    print("🧠 Risveglio della Rete Neurale...")
    model = tf.keras.models.load_model(MODEL_NAME, compile=False)
    
    # Dizionario per salvare i risultati (MAE) per ogni livello di rumore
    risultati_mae = {snr: [] for snr in SNR_LEVELS}
    
    print("🔮 Iniezione del rumore ed elaborazione...")
    for snr in SNR_LEVELS:
        print(f"\n--- Analisi livello SNR: {snr}{' dB' if snr != 'Clean' else ''} ---")
        X_batch = []
        y_batch = []
        
        for index, row in tqdm(df_test.iterrows(), total=len(df_test)):
            audio_path = os.path.join(DIR_WET, row['wet_audio_file'])
            if not os.path.exists(audio_path): continue
                
            try:
                y, sr = librosa.load(audio_path, sr=16000)
                
                # Aggiungiamo il rumore solo se non siamo nel caso 'Clean'
                if snr != 'Clean':
                    y = mixa_rumore(y, snr_db=snr)
                    
                spec = estrai_feature_al_volo(y, sr)
                X_batch.append(spec)
                y_batch.append([row['t30_1000Hz'], row['t30_2000Hz'], row['t30_4000Hz']])
            except Exception as e:
                continue
                
        X_batch = np.array(X_batch)[..., np.newaxis]
        y_batch = np.array(y_batch)
        
        # Facciamo indovinare la rete
        y_pred = model.predict(X_batch, verbose=0)
        
        # Calcoliamo l'errore medio per le tre frequenze e salviamolo
        mae_1000 = np.mean(np.abs(y_batch[:, 0] - y_pred[:, 0]))
        mae_2000 = np.mean(np.abs(y_batch[:, 1] - y_pred[:, 1]))
        mae_4000 = np.mean(np.abs(y_batch[:, 2] - y_pred[:, 2]))
        
        risultati_mae[snr] = [mae_1000, mae_2000, mae_4000]

    # --- 4. CREAZIONE GRAFICO ---
    print("\n📊 Creazione del Grafico...")
    
    etichette_x = [str(snr) + (" dB" if snr != "Clean" else "") for snr in SNR_LEVELS]
    
    val_1000 = [risultati_mae[snr][0] for snr in SNR_LEVELS]
    val_2000 = [risultati_mae[snr][1] for snr in SNR_LEVELS]
    val_4000 = [risultati_mae[snr][2] for snr in SNR_LEVELS]
    
    plt.figure(figsize=(10, 6))
    plt.plot(etichette_x, val_1000, marker='o', linewidth=3, markersize=8, label='1000 Hz')
    plt.plot(etichette_x, val_2000, marker='s', linewidth=3, markersize=8, label='2000 Hz')
    plt.plot(etichette_x, val_4000, marker='^', linewidth=3, markersize=8, label='4000 Hz')
    
    # Invertiamo l'asse X per far vedere il degrado da Sinistra (Pulito) a Destra (Molto Rumoroso)
    plt.gca().invert_xaxis() 
    
    plt.title('Sensibilità al Rumore Rosa (Noise Robustness)', fontsize=16, fontweight='bold')
    plt.xlabel('Livello di Rumore Aggiunto (SNR)', fontsize=12)
    plt.ylabel('Errore Medio Assoluto - MAE (secondi)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    salvataggio = "degrado_rumore_XR.png"
    plt.savefig(salvataggio, dpi=300)
    print(f"✅ Grafico salvato come '{salvataggio}'.")
    plt.show()

if __name__ == "__main__":
    main()