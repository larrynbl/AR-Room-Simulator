import os
import numpy as np
import librosa
import tensorflow as tf
import warnings

# Disabilita i noiosi warning di TensorFlow per avere un output pulito da mostrare
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# --- CONFIGURAZIONE ---
# Usiamo il modello finale adattato al mondo reale
MODEL_PATH = os.path.join('data', 't30_global_adapted_model.h5')
MAX_FRAMES = 1300

def stima_riverbero(audio_path):
    """
    Simula il backend di un'app AR: riceve un audio, lo analizza e restituisce il T30.
    """
    if not os.path.exists(audio_path):
        return f"Errore: File '{audio_path}' non trovato."

    print(f"🎙️ Analisi acustica dell'ambiente in corso...")
    print(f"📂 File: {os.path.basename(audio_path)}")
    
    try:
        # 1. Caricamento Audio e Preprocessing (come in addestramento)
        y, sr = librosa.load(audio_path, sr=44100, mono=True)
        
        # 2. Estrazione Spettrogramma
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # 3. Padding/Cropping
        if S_dB.shape[1] > MAX_FRAMES:
            S_dB = S_dB[:, :MAX_FRAMES]
        else:
            S_dB = np.pad(S_dB, pad_width=((0,0), (0, MAX_FRAMES - S_dB.shape[1])), mode='constant')
            
        X_input = S_dB[np.newaxis, ..., np.newaxis]
        
        # 4. Caricamento Modello e Predizione
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        pred_t30 = model.predict(X_input, verbose=0)[0][0]
        
        return float(pred_t30)
        
    except Exception as e:
        return f"Errore durante l'elaborazione: {e}"

# ==========================================
# TEST DEL SISTEMA
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print(" 🔊 AR ACOUSTIC ENGINE - INIZIALIZZAZIONE")
    print("="*50)
    
    # ⚠️ INSERISCI QUI IL NOME DI UN FILE AUDIO PER FARE UN TEST
    # Puoi usare uno dei file dentro data/real_recs_clean/
    file_da_testare = os.path.join('data', 'real_recs', 'test_d_aula.wav') 
    
    risultato = stima_riverbero(file_da_testare)
    
    if isinstance(risultato, float):
        print("\n✅ RISULTATO DELL'INTELLIGENZA ARTIFICIALE:")
        print(f"   ► T30 Stimato: {risultato:.3f} secondi")
        print("   Il parametro è pronto per essere inviato al motore AR (Unity/FMOD).")
    else:
        print(risultato)
    print("="*50)