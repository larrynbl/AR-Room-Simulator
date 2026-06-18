import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

CSV_CLEAN = "dataset_ml_clean.csv"
DIR_WET = "dataset_wet"
DIR_SPEC = "dataset_spectrograms"

os.makedirs(DIR_SPEC, exist_ok=True)

def main():
    print("🧠 Avvio Estrazione Feature (Allineamento + Z-Score)...")
    
    try:
        df = pd.read_csv(CSV_CLEAN)
    except FileNotFoundError:
        print(f"❌ Errore: {CSV_CLEAN} non trovato.")
        return

    for index, row in tqdm(df.iterrows(), total=len(df)):
        audio_file = row['wet_audio_file']
        audio_path = os.path.join(DIR_WET, audio_file)

        if not os.path.exists(audio_path):
            continue

        try:
            # 1. Carica a 16kHz
            y, sr = librosa.load(audio_path, sr=16000)
            
            # 2. FIX CRITICO 1: Trova l'inizio della voce e SALVA LA CODA
            # Split restituisce gli intervalli [inizio, fine] in cui c'è suono.
            intervals = librosa.effects.split(y, top_db=40)
            if len(intervals) > 0:
                start_sample = intervals[0][0] # Il primissimo istante in cui inizia a parlare
                # Taglia via solo il silenzio iniziale (da start_sample fino alla fine dell'array)
                y_aligned = y[start_sample:]
            else:
                y_aligned = y

            # 3. Alta Risoluzione Temporale (hop_length = 256)
            mel_spec = librosa.feature.melspectrogram(y=y_aligned, sr=sr, n_fft=2048, hop_length=256, n_mels=128)

            # 4. Aumento del Contrasto in Decibel
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)

            # 5. FIX CRITICO 2: NORMALIZZAZIONE Z-SCORE PER SINGOLA IMMAGINE
            # Centra i dati portando la media a 0 e la deviazione standard a 1.
            # Questo fa "esplodere" il contrasto visivo della coda di riverbero.
            mean_val = np.mean(mel_spec_db)
            std_val = np.std(mel_spec_db)
            mel_spec_norm = (mel_spec_db - mean_val) / (std_val + 1e-8) # 1e-8 evita la divisione per zero in caso di silenzio assoluto

            spec_filename = audio_file.replace('.wav', '.npy')
            spec_path = os.path.join(DIR_SPEC, spec_filename)
            
            np.save(spec_path, mel_spec_norm.astype(np.float32))

        except Exception as e:
            print(f"\n ❌ Errore processando {audio_file}: {e}")

if __name__ == "__main__":
    main()