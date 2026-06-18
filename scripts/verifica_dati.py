import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE ---
CSV_CLEAN = "dataset_ml_clean.csv"
DIR_SPEC = "dataset_spectrograms"

def main():
    print("🔬 Avvio ispezione visiva del Dataset...")
    
    try:
        df = pd.read_csv(CSV_CLEAN)
    except FileNotFoundError:
        print(f"❌ Errore: {CSV_CLEAN} non trovato.")
        return

    # Scegliamo 6 tracce audio a caso dal tuo dataset
    campioni = df.sample(n=6)
    
    # Prepariamo una "tela" con 6 riquadri (2 righe x 3 colonne)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    
    print("Generazione dei grafici...")

    for idx, (index, row) in enumerate(campioni.iterrows()):
        audio_file = row['wet_audio_file']
        spec_filename = audio_file.replace('.wav', '.npy')
        spec_path = os.path.join(DIR_SPEC, spec_filename)

        if not os.path.exists(spec_path):
            axes[idx].set_title(f"File non trovato:\n{spec_filename}")
            continue

        # Carichiamo la matrice Z-Score
        spec = np.load(spec_path)
        
        # Disegniamo l'immagine. 
        # cmap='magma' è ottima perché il nero è il silenzio e il giallo/bianco è l'energia
        ax = axes[idx]
        img = ax.imshow(spec, aspect='auto', origin='lower', cmap='magma')
        
        # Scriviamo il T30 reale come titolo per fare il confronto
        t30 = row['t30_broadband']
        tipo = row['room_type']
        ax.set_title(f"[{tipo.upper()}] T30: {t30:.2f} s", fontsize=12, fontweight='bold')
        ax.set_xlabel("Tempo (Frames)")
        ax.set_ylabel("Frequenze (Bande Mel)")
        
        # Barra del colore (i valori saranno centrati attorno allo 0 per via dello Z-Score)
        fig.colorbar(img, ax=ax)

    plt.tight_layout()
    print("✅ Guarda la finestra aperta per ispezionare i dati!")
    plt.show()

if __name__ == "__main__":
    main()