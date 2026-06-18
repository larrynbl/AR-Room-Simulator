import pandas as pd
import numpy as np

# --- CONFIGURAZIONE ---
LOG_WET_IN = "master_log_wet.csv"         # Il file "sporco" in ingresso
LOG_CLEAN_OUT = "dataset_ml_clean.csv"    # Il file "pulito" in uscita

def main():
    print("🧹 Avvio pulizia del dataset...")
    
    # 1. Lettura intelligente (auto-detect del separatore)
    try:
        df = pd.read_csv(LOG_WET_IN, sep=None, engine='python')
    except Exception as e:
        print(f"❌ Errore nella lettura del file: {e}")
        return

    righe_iniziali = len(df)
    print(f"📊 Righe lette dal file: {righe_iniziali}")

    # 2. Igienizzazione dei numeri (da formato EU a formato Standard ML)
    colonne_t30 = ['t30_1000Hz', 't30_2000Hz', 't30_4000Hz', 't30_broadband']
    
    for col in colonne_t30:
        if col in df.columns:
            # Se la colonna è testo (es. "1,54"), sostituiamo la virgola col punto
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.')
            
            # Forziamo la conversione in numero reale (float). 
            # errors='coerce' trasforma eventuali testi strani irrecuperabili in "NaN" (vuoto)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Eliminiamo le righe che contengono valori NaN nei T30
    df = df.dropna(subset=colonne_t30)
    print(f"🛠️ Righe valide dopo correzione formattazione: {len(df)}")

    # 3. Taglio degli Outlier (Scartiamo T30 > 5 secondi o <= 0)
    # Teniamo solo le stanze dove il broadband è logico per la XR (tra 0.01 e 5.0 secondi)
    df_clean = df[(df['t30_broadband'] > 0.0) & (df['t30_broadband'] <= 5.0)]

    stanze_scartate = len(df) - len(df_clean)
    
    print(f"✂️ Scartati {stanze_scartate} outlier (T30 troppo alti o innaturali).")
    print(f"✅ DATASET PRONTO: {len(df_clean)} tracce audio valide.")

    # 4. Salvataggio formattato in Standard Internazionale
    # sep=',' e decimal='.' garantiscono che PyTorch/Keras lo leggeranno al 100% senza errori
    df_clean.to_csv(LOG_CLEAN_OUT, index=False, sep=',', decimal='.')
    print(f"💾 File pulito salvato in: {LOG_CLEAN_OUT}")

if __name__ == "__main__":
    main()