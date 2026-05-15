import whisper
import os

def esegui_trascrizione(file_audio):
    # Controlliamo se il file esiste prima di iniziare
    if not os.path.exists(file_audio):
        print(f"Errore: Il file '{file_audio}' non è stato trovato.")
        return

    print("--- Inizializzazione ---")
    # Carichiamo il modello. 'base' è il miglior compromesso tra velocità e precisione.
    # Se hai una buona connessione e PC potente, puoi usare 'medium'.
    model = whisper.load_model("base")

    print(f"--- Inizio elaborazione di: {file_audio} ---")
    print("Nota: Vedrai il testo apparire man mano che viene decodificato.\n")

    try:
        # La funzione transcribe trasforma l'audio in testo.
        # verbose=True ti permette di vedere i progressi riga per riga.
        result = model.transcribe(file_audio, verbose=True)

        # Salvataggio del risultato in un file .txt
        nome_file_txt = os.path.splitext(file_audio)[0] + ".txt"
        with open(nome_file_txt, "w", encoding="utf-8") as f:
            f.write(result["text"])

        print("\n--- Operazione completata! ---")
        print(f"Il testo è stato salvato in: {nome_file_txt}")
        
    except Exception as e:
        print(f"\nSi è verificato un errore durante la trascrizione: {e}")

# --- CONFIGURAZIONE ---
# Inserisci qui il nome del tuo file audio (es. "lezione.mp3" o "intervista.wav")
nome_file_da_elaborare = r"C:\Users\lolle\Desktop\uni\CAPSTONE\data\40min\speech_2.mp4"

if __name__ == "__main__":
    esegui_trascrizione(nome_file_da_elaborare)