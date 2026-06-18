import os
import numpy as np
import tensorflow as tf
import librosa

# --- CONFIGURAZIONE ---
DIR_REALE = "test_reale"           # La cartella con le tue registrazioni
MODEL_NAME = "modello_xr_t30.h5"   # Il tuo modello addestrato
TARGET_FRAMES = 512

def estrai_feature_al_volo(y, sr=16000):
    """La nostra pipeline standard di estrazione e normalizzazione"""
    # 1. Troviamo dove inizia l'audio utile (Voice Activity)
    intervals = librosa.effects.split(y, top_db=40)
    if len(intervals) > 0:
        y_aligned = y[intervals[0][0]:]
    else:
        y_aligned = y

    # 2. Spettrogramma ad alta risoluzione
    mel_spec = librosa.feature.melspectrogram(y=y_aligned, sr=sr, n_fft=2048, hop_length=256, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)

    # 3. Normalizzazione Z-Score per singola immagine
    mean_val = np.mean(mel_spec_db)
    std_val = np.std(mel_spec_db)
    norm_spec = (mel_spec_db - mean_val) / (std_val + 1e-8)

    # 4. Padding/Trimming a 512 frame
    if norm_spec.shape[1] < TARGET_FRAMES:
        pad_width = TARGET_FRAMES - norm_spec.shape[1]
        norm_spec = np.pad(norm_spec, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=-5.0)
    else:
        norm_spec = norm_spec[:, :TARGET_FRAMES]
        
    return norm_spec

def main():
    print("🎤 Avvio Test Sim-to-Real (Registrazioni Reali .OGG)...")
    
    if not os.path.exists(DIR_REALE):
        os.makedirs(DIR_REALE)
        print(f"⚠️ Cartella '{DIR_REALE}' creata. Inserisci i tuoi file audio e riavvia.")
        return
        
    # FIX: Aggiunto '.ogg' alla tupla delle estensioni accettate
    file_audio = [f for f in os.listdir(DIR_REALE) if f.endswith(('.wav', '.flac', '.mp3', '.m4a', '.ogg'))]
    
    if not file_audio:
        print(f"❌ Nessun file audio trovato in '{DIR_REALE}'. Assicurati di aver messo i file .ogg lì dentro.")
        return

    print("🧠 Risveglio della Rete Neurale...\n")
    model = tf.keras.models.load_model(MODEL_NAME, compile=False)

    for file in file_audio:
        path = os.path.join(DIR_REALE, file)
        try:
            # Librosa carica ed effettua il resampling a 16kHz in automatico anche per gli .ogg
            y, sr = librosa.load(path, sr=16000)
            
            # Estraiamo la feature
            spec = estrai_feature_al_volo(y, sr)
            
            # Adattiamo la forma per Keras: (1 batch, 128 mel, 512 frames, 1 canale)
            X_input = spec[np.newaxis, ..., np.newaxis]
            
            # Inferenza
            predizione = model.predict(X_input, verbose=0)[0]
            
            print(f"🎧 File Analizzato: {file}")
            print(f"   ➔ T30 stimato a 1000 Hz: {predizione[0]:.3f} s")
            print(f"   ➔ T30 stimato a 2000 Hz: {predizione[1]:.3f} s")
            print(f"   ➔ T30 stimato a 4000 Hz: {predizione[2]:.3f} s")
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Errore processando {file}: {e}")

if __name__ == "__main__":
    main()