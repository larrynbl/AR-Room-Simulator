import os
import random
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import fftconvolve

# --- CONFIGURAZIONE ---
DIR_RIR = "dataset_audio"              # Dove hai salvato le tue RIR
DIR_DRY = "data/dry_voices"            # Dove hai i file .flac o .wav puliti
DIR_WET = "dataset_wet"                # Dove salveremo l'audio finale
LOG_RIR = "master_log_t30.csv"         # Il CSV con i T30 generato prima
LOG_WET = "master_log_wet.csv"         # Il nuovo CSV per il Machine Learning

# --- NUOVI PARAMETRI DSP (Selezione Pause) ---
MIN_SILENCE_RATIO = 0.30  # Esige almeno il 30% di silenzio/pause nella traccia
SILENCE_THRESH_DB = 35    # Quanti decibel sotto il picco consideriamo come "silenzio"

os.makedirs(DIR_WET, exist_ok=True)

def main():
    print("🚀 Avvio Motore di Convoluzione (Filtro VAD attivo per le pause)...")
    
    # 1. Carichiamo i dati delle stanze
    if not os.path.exists(LOG_RIR):
        print(f"❌ Errore: File {LOG_RIR} non trovato.")
        return
        
    df_rir = pd.read_csv(LOG_RIR, sep=None, engine='python')
    
    dry_files = []
    for root, _, files in os.walk(DIR_DRY):
        for file in files:
            if file.endswith((".wav", ".flac")):
                dry_files.append(os.path.join(root, file))
                
    if not dry_files:
        print(f"❌ Errore: Nessun file audio Dry trovato in {DIR_DRY}")
        return
        
    print(f"✅ Trovate {len(df_rir)} RIR valide e {len(dry_files)} voci Dry.")
    
    log_wet = []
    scartati_cache = set() # Memoria per non ri-analizzare i file senza pause
    
    # 2. Per ogni RIR, creiamo UNA traccia riverberata
    for index, row in df_rir.iterrows():
        rir_path = os.path.normpath(os.path.join(DIR_RIR, row['file_audio']))
        
        if pd.isna(row['t30_1000Hz']):
            continue
            
        if not os.path.exists(rir_path):
            continue

        # --- IL "PROVINO" (Cerchiamo l'audio con le pause giuste) ---
        dry_valido = None
        tentativi = 0
        ratio_silenzio = 0
        
        # Facciamo max 20 tentativi per stanza, per evitare loop infiniti
        while dry_valido is None and tentativi < 20:
            candidato_dry = random.choice(dry_files)
            
            # Se lo abbiamo già scartato in passato, saltiamolo subito
            if candidato_dry in scartati_cache:
                tentativi += 1
                continue
                
            try:
                # Carichiamo in modo ultra-rapido (sr=None mantiene la freq originale per fare in fretta)
                y_test, _ = librosa.load(candidato_dry, sr=None)
                
                # Troviamo tutti gli intervalli in cui la persona sta parlando
                intervalli_attivi = librosa.effects.split(y_test, top_db=SILENCE_THRESH_DB)
                
                # Sommiamo la durata di tutte le parole dette
                campioni_parlati = np.sum([end - start for start, end in intervalli_attivi])
                campioni_totali = len(y_test)
                
                # Calcoliamo la percentuale di silenzio (le pause tra le parole + inizio/fine)
                ratio_silenzio = 1.0 - (campioni_parlati / campioni_totali)
                
                if ratio_silenzio >= MIN_SILENCE_RATIO:
                    dry_valido = candidato_dry  # Trovato! Ha abbastanza pause!
                else:
                    scartati_cache.add(candidato_dry) # È un "muro di parole", scartiamolo per sempre
                    tentativi += 1
                    
            except Exception:
                scartati_cache.add(candidato_dry)
                tentativi += 1

        # Se dopo 20 tentativi non troviamo nulla, passiamo oltre
        if dry_valido is None:
            print(f" ⚠️ Nessun audio adatto per {os.path.basename(rir_path)}. Salto.")
            continue
            
        try:
            # 3. Lettura finale e Convoluzione (Ora sappiamo che la Dry è perfetta)
            rir, fs_rir = sf.read(rir_path)
            
            # Ricampioniamo in modo sicuro alla frequenza della stanza
            dry, _ = librosa.load(dry_valido, sr=fs_rir)
            
            if rir.ndim > 1: rir = rir[:, 0]
            if dry.ndim > 1: dry = dry[:, 0]
            
            # 4. Convoluzione
            wet_audio = fftconvolve(dry, rir, mode='full')
            
            max_val = np.max(np.abs(wet_audio))
            if max_val > 0:
                wet_audio = wet_audio / max_val
            
            # 5. Salvataggio
            nome_stanza = os.path.basename(rir_path).replace('.wav', '')
            nome_wet = f"wet_{nome_stanza}.wav"
            out_path = os.path.join(DIR_WET, nome_wet)
            
            sf.write(out_path, wet_audio, fs_rir)
            
            log_wet.append({
                "wet_audio_file": nome_wet,
                "source_dry": os.path.basename(dry_valido),
                "source_rir": row['file_audio'],
                "room_type": row['room_type'],
                "t30_1000Hz": row['t30_1000Hz'],
                "t30_2000Hz": row['t30_2000Hz'],
                "t30_4000Hz": row['t30_4000Hz'],
                "t30_broadband": row['t30_broadband']
            })
            
            # Stampiamo anche la percentuale di pause per avere un feedback visivo
            print(f" 🎧 Creado: {nome_wet} (Pause: {ratio_silenzio*100:.1f}%)")
            
        except Exception as e:
            print(f" ⚠️ Errore convoluzione di {rir_path}: {e}")

    pd.DataFrame(log_wet).to_csv(LOG_WET, index=False)
    print(f"\n🎉 FASE 2 COMPLETATA! Scartati {len(scartati_cache)} file Dry perché troppo densi.")
    print(f"Dati salvati e pronti per il ML in {LOG_WET}")

if __name__ == "__main__":
    main()