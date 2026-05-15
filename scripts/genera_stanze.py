"""
genera_dataset_xr.py
====================
Script universale per la generazione del dataset sintetico
AI-Driven XR Acoustic Simulator — Fase 1/3.

Iterazione su file .xlsx in INPUT_DIR: per ogni stanza ShoeBox genera la RIR
con pyroomacoustics (ray tracing) e ne estrae il T30 su 6 bande d'ottava
(125 Hz – 4 kHz) tramite pipeline DSP scipy.signal (modulo dsp_octave_t30).

Modifiche rispetto alla versione broadband originale
-----------------------------------------------------
  [A] Import di extract_t30_per_band da dsp_octave_t30
  [B] La RIR grezza viene passata al modulo DSP *prima* della normalizzazione
      (la normalizzazione altera l'integrale di Schroeder e quindi il T30)
  [C] log_totale raccoglie 7 colonne T30 invece di una sola

Tutto il resto — safe_float, crea_materiale, ShoeBox, ray tracing,
set_ray_tracing, salvataggio WAV, gc.collect — è invariato.
"""

import os
import sys
import glob
import logging

import numpy as np
import pandas as pd
import pyroomacoustics as pra
import soundfile as sf
import gc

# [A] Import del modulo DSP per-banda.
# Aggiunge la directory dello script al path di ricerca dei moduli,
# così l'import funziona indipendentemente da dove viene lanciato lo script
# (es. "python scripts/genera_stanze.py" dalla root del progetto).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsp_octave_t30 import extract_t30_per_band, BAND_LABELS

# ---------------------------------------------------------------------------
# Logging — messaggi di warning del modulo DSP visibili in console
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s"
)

# ---------------------------------------------------------------------------
# 1. CONFIGURAZIONE
# ---------------------------------------------------------------------------
INPUT_DIR  = "input_csv"
OUTPUT_DIR = "dataset_audio"
LOG_FILE   = "master_log_t30.csv"
SAMPLE_RATE = 44100

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. FUNZIONI ESTRATTORE MATERIALI (invariate)
# ---------------------------------------------------------------------------

def safe_float(val):
    try:
        v = float(str(val).replace(',', '.'))
        return 0.1 if np.isnan(v) else v
    except:
        return 0.1


def crea_materiale(row, prefix):
    c_125 = safe_float(row[f'{prefix}_a125'])
    c_250 = safe_float(row[f'{prefix}_a250'])
    c_500 = safe_float(row[f'{prefix}_a500'])
    c_1k  = safe_float(row[f'{prefix}_a1000'])
    c_2k  = safe_float(row[f'{prefix}_a2000'])
    c_4k  = safe_float(row[f'{prefix}_a4000'])

    coeffs = [c_125, c_250, c_500, c_1k, c_2k, c_4k]
    coeffs_puliti = [float(x) for x in np.clip(coeffs, 0.02, 0.99)]

    return pra.Material(
        energy_absorption={
            "description": f"Custom_{prefix}",
            "coeffs": coeffs_puliti,
            "center_freqs": [125, 250, 500, 1000, 2000, 4000],
        },
        scattering=0.2,
    )


# ---------------------------------------------------------------------------
# 3. MOTORE PRINCIPALE
# ---------------------------------------------------------------------------

def main():
    excel_files = glob.glob(os.path.join(INPUT_DIR, "*.xlsx"))

    if not excel_files:
        print(f"❌ Nessun file XLSX trovato nella cartella '{INPUT_DIR}'.")
        return

    print(f"🚀 Trovati {len(excel_files)} file Excel da processare.\n")
    log_totale = []

    for file_excel in excel_files:
        nome_base = os.path.basename(file_excel).replace('.xlsx', '')
        print(f"--- Processando: {nome_base} ---")

        out_sub_dir = os.path.join(OUTPUT_DIR, nome_base)
        os.makedirs(out_sub_dir, exist_ok=True)

        try:
            df = pd.read_excel(file_excel)
        except Exception as e:
            print(f"❌ Errore nella lettura del file {file_excel}: {e}")
            continue

        for index, row in df.iterrows():
            variant_id = row['variant_id']
            room_type  = row['room_type']

            try:
                # --- Geometria (invariata) ---
                L, W, H = float(row['L_m']), float(row['W_m']), float(row['H_m'])
                src_pos = [float(row['src_x']), float(row['src_y']), float(row['src_z'])]
                mic_pos = [float(row['mic_x']), float(row['mic_y']), float(row['mic_z'])]

                # --- Materiali ---
                mats = {
                    "floor":   crea_materiale(row, 'floor'),
                    "ceiling": crea_materiale(row, 'ceiling'),
                    "east":    crea_materiale(row, 'wall_east'),
                    "west":    crea_materiale(row, 'wall_west'),
                    "north":   crea_materiale(row, 'wall_north'),
                    "south":   crea_materiale(row, 'wall_south'),
                }

                # --- Stanza e ray tracing (invariati) ---
                room = pra.ShoeBox(
                    [L, W, H], materials=mats, fs=SAMPLE_RATE, max_order=15,
                    ray_tracing=True, air_absorption=True
                )
                room.set_ray_tracing(receiver_radius=0.1, n_rays=10000, energy_thres=1e-5)
                room.add_source(src_pos)
                room.add_microphone(mic_pos)

                # --- Calcolo RIR con patch del bug PRA broadcast ---
                # Il bug "operands could not be broadcast together (N,)(N+1,)"
                # si manifesta dentro compute_rir() quando gli array ISM e ray
                # tracing risultano disallineati di esattamente 1 sample.
                # La causa è in room.py:2678 dove numpy tenta di sommare due
                # array di lunghezza diversa di 1 elemento.
                #
                # Strategia di fix: dopo compute_rir(), allineare manualmente
                # tutte le RIR del mic al array più lungo tra ISM e ray tracing
                # tramite zero-padding. Questo è l'approccio meno invasivo:
                # non tocca la fisica della simulazione, aggiusta solo il buffer.
                try:
                    room.compute_rir()
                except ValueError as rir_err:
                    if "operands could not be broadcast" not in str(rir_err):
                        raise  # errore diverso → propagato al try/except esterno
                    # Patch: forza la lunghezza di tutti i canali RIR al massimo
                    # tra ISM e ray tracing con zero-padding dell'array più corto.
                    # room.rir è una lista [mic_idx][src_idx] → array 1D
                    _rir_ism = room.rir_ISM if hasattr(room, 'rir_ISM') else None
                    _rir_rt  = room.rir_rt60 if hasattr(room, 'rir_rt60') else None
                    if _rir_ism is not None and _rir_rt is not None:
                        for _m in range(len(room.rir)):
                            for _s in range(len(room.rir[_m])):
                                _a = _rir_ism[_m][_s]
                                _b = _rir_rt[_m][_s]
                                _n = max(len(_a), len(_b))
                                _a2 = np.pad(_a, (0, _n - len(_a)))
                                _b2 = np.pad(_b, (0, _n - len(_b)))
                                room.rir[_m][_s] = _a2 + _b2
                        print(f"  ↩️  Stanza {variant_id}: bug broadcast riparato con zero-pad")
                    else:
                        # Fallback finale: disabilita ray tracing per evitare il bug
                        # di broadcast ISM+RT. La modalità ISM pura non ha il combining
                        # step che genera il mismatch di ±1 sample.
                        room = pra.ShoeBox(
                            [L, W, H], materials=mats, fs=SAMPLE_RATE, max_order=15,
                            ray_tracing=False, air_absorption=True
                        )
                        room.add_source(src_pos)
                        room.add_microphone(mic_pos)
                        room.compute_rir()
                        print(f"  ↩️  Stanza {variant_id}: fallback ray_tracing=False applicato")

                # [B] RIR grezza estratta PRIMA della normalizzazione:
                #     extract_t30_per_band lavora sull'energia assoluta del segnale;
                #     normalizzare prima altera l'integrale di Schroeder e
                #     renderebbe i T30 per-banda fisicamente incorretti.
                rir_raw = room.rir[0][0].copy()

                # --- Estrazione T30 per-banda (DSP) ---
                #     Sostituisce: t30_globale = float(room.measure_rt60()[0][0])
                #     Il try/except interno al modulo gestisce banda per banda,
                #     incluso il bug "operands could not be broadcast together".
                t30_dict = extract_t30_per_band(rir_raw, fs=float(SAMPLE_RATE))

                # --- Normalizzazione e salvataggio WAV (invariati) ---
                rir_norm = rir_raw / np.max(np.abs(rir_raw))
                nome_audio = f"{room_type}_{int(variant_id):03d}.wav"
                path_audio = os.path.join(out_sub_dir, nome_audio)
                sf.write(path_audio, rir_norm, SAMPLE_RATE)

                # [C] Log con 7 colonne T30 invece della singola t30_target_globale
                log_entry = {
                    "file_audio": f"{nome_base}/{nome_audio}",
                    "room_type":  room_type,
                    "volume_m3":  row['V_m3'],
                    # Colonne per-banda dal modulo DSP
                    **{k: round(v, 4) if not np.isnan(v) else np.nan
                       for k, v in t30_dict.items()},
                }
                log_totale.append(log_entry)

                # Stampa di avanzamento — mostra t30_broadband come riferimento rapido
                t30_bb = t30_dict.get("t30_broadband", np.nan)
                t30_str = f"{t30_bb:.3f}s" if not np.isnan(t30_bb) else "NaN"
                print(f"  ✅ Stanza {variant_id} | broadband T30: {t30_str}")

            except Exception as e:
                print(f"  ⚠️ Errore alla stanza ID {variant_id}: {e}")

        gc.collect()
        print(f"--- Completato: {nome_base} ---\n")

    # --- Salvataggio CSV finale ---
    # Ordine colonne: metadati → bande per-banda → broadband
    col_order = (
        ["file_audio", "room_type", "volume_m3"]
        + BAND_LABELS          # t30_125 … t30_4k
        + ["t30_broadband"]
    )
    df_log = pd.DataFrame(log_totale)

    # Aggiunge colonne mancanti (es. se tutte le stanze di un file falliscono una banda)
    for col in col_order:
        if col not in df_log.columns:
            df_log[col] = np.nan

    # sep=';' e decimal=',' producono il formato CSV italiano nativo:
    # Excel su Windows con locale italiano apre il file direttamente
    # senza wizard e senza spostamenti decimali.
    df_log[col_order].to_csv(LOG_FILE, index=False, decimal=',', sep=';')
    print(f"🎉 TUTTO COMPLETATO! Log master salvato in '{LOG_FILE}'")
    print(f"   Colonne T30: {BAND_LABELS + ['t30_broadband']}")


if __name__ == "__main__":
    main()