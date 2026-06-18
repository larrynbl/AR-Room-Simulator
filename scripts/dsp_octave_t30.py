import numpy as np
from scipy.signal import butter, sosfilt
import logging

# Teniamo SOLO le bande dai 1000 Hz in su, come concordato per la nuova strategia
OCTAVE_BANDS = [
    (1000, 707.11, 1414.21),
    (2000, 1414.21, 2828.43),
    (4000, 2828.43, 5656.85)
]

def calculate_t30_from_rir(rir_signal, fs):
    """Calcola il T30 da una RIR usando l'integrale di Schroeder."""
    energy = rir_signal ** 2
    # Schroeder backward integration
    schroeder = np.cumsum(energy[::-1])[::-1]
    
    # Normalizzazione in dB
    schroeder = np.maximum(schroeder, np.finfo(float).eps)
    schroeder_db = 10.0 * np.log10(schroeder / np.max(schroeder))

    try:
        # Trova i punti a -5 dB e -35 dB
        idx_minus_5 = np.where(schroeder_db <= -5.0)[0][0]
        idx_minus_35 = np.where(schroeder_db <= -35.0)[0][0]
    except IndexError:
        return np.nan 

    time_axis = np.arange(len(schroeder_db)) / fs
    t_5 = time_axis[idx_minus_5]
    t_35 = time_axis[idx_minus_35]

    # Formula ISO 3382 per T30: moltiplica il decadimento di 30dB per 2
    t30 = 2.0 * (t_35 - t_5)
    return float(t30)

def extract_t30_per_band(rir_signal, fs=44100):
    """Filtra la RIR nelle 3 bande alte ed estrae i T30."""
    results = {}

    for f_center, f_low, f_high in OCTAVE_BANDS:
        # Filtro IIR passa-banda standard (Butterworth)
        sos = butter(N=6, Wn=[f_low, f_high], btype='bandpass', fs=fs, output='sos')
        filtered_rir = sosfilt(sos, rir_signal)
        
        t30_band = calculate_t30_from_rir(filtered_rir, fs)
        results[f"t30_{f_center}"] = round(t30_band, 3) if not np.isnan(t30_band) else None

    # Calcolo Broadband globale (senza filtri)
    t30_broadband = calculate_t30_from_rir(rir_signal, fs)
    results["t30_broadband"] = round(t30_broadband, 3) if not np.isnan(t30_broadband) else None

    return results