import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURAZIONE ---
CSV_CLEAN = "dataset_ml_clean.csv"
DIR_SPEC = "dataset_spectrograms"
MODEL_NAME = "modello_xr_t30.h5"

# Fissiamo la larghezza dello spettrogramma. 
# 512 frame corrispondono a circa 16 secondi di audio, perfetti per catturare la coda del riverbero.
TARGET_FRAMES = 512

# Mettila così nel tuo train_cnn.py:
def load_and_pad_data():
    print("🔄 Caricamento degli spettrogrammi normalizzati nella RAM...")
    df = pd.read_csv(CSV_CLEAN)
    X, y = [], []
    for index, row in df.iterrows():
        spec_filename = row['wet_audio_file'].replace('.wav', '.npy')
        spec_path = os.path.join(DIR_SPEC, spec_filename)
        if not os.path.exists(spec_path): continue
            
        spec = np.load(spec_path) 
        
        TARGET_FRAMES = 512
        if spec.shape[1] < TARGET_FRAMES:
            # Riempiamo gli spazi vuoti con -80.0 (che dopo lo Z-score corrisponde a valori molto negativi)
            pad_width = TARGET_FRAMES - spec.shape[1]
            spec = np.pad(spec, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=-5.0) 
        else:
            spec = spec[:, :TARGET_FRAMES]
            
        X.append(spec)
        y.append([row['t30_1000Hz'], row['t30_2000Hz'], row['t30_4000Hz']])
        
    X = np.array(X)[..., np.newaxis] 
    y = np.array(y)
    print(f"✅ Forma input (X): {X.shape} | Forma target (y): {y.shape}")
    return X, y

def build_model(input_shape):
    print("🧠 Costruzione della CNN (Architettura Asimmetrica)...")
    model = models.Sequential()
    
    # BLOCCO 1: Filtro Asimmetrico (3, 7) per guardare LUNGO nel tempo
    model.add(layers.Conv2D(32, (3, 7), input_shape=input_shape, padding='same'))
    model.add(layers.BatchNormalization()) # Stabilizza il segnale
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 3))) # Riduciamo di più il tempo rispetto alla frequenza
    
    # BLOCCO 2: Coda di riverbero intermedia
    model.add(layers.Conv2D(64, (3, 5), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # BLOCCO 3: Dettagli finali
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    
    model.add(layers.Dense(128))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.4)) 
    
    # STRATO DI OUTPUT
    model.add(layers.Dense(3, activation='linear'))
    
    # Usiamo Huber Loss: è molto più robusta della MSE per i valori estremi del riverbero
    model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=['mae'])
    return model

def main():
    X, y = load_and_pad_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model(input_shape=(X.shape[1], X.shape[2], 1))
    
    checkpoint = callbacks.ModelCheckpoint(MODEL_NAME, save_best_only=True, monitor='val_mae', mode='min')
    early_stop = callbacks.EarlyStopping(monitor='val_mae', patience=15, restore_best_weights=True)
    
    # TRUCCO 3: Riduttore del Learning Rate
    # Se per 5 epoche l'errore non scende, il modello "rallenta" per studiare meglio
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    
    print("\n🚀 INIZIO ADDESTRAMENTO OTTIMIZZATO...")
    history = model.fit(
        X_train, y_train,
        epochs=80, # Aumentiamo le epoche perché abbiamo l'Early Stop e l'LR dinamico
        batch_size=16, 
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stop, lr_scheduler]
    )
    
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n🏆 RISULTATO FINALE: Errore Medio Assoluto (MAE) = {mae:.3f} secondi!")

if __name__ == "__main__":
    main()