import os
import glob
import numpy as np
import pandas as pd
import pyroomacoustics as pra
import soundfile as sf
import gc 

# Importiamo il nostro estrattore DSP personalizzato
from dsp_octave_t30 import extract_t30_per_band

# --- 1. CONFIGURAZIONE ---
INPUT_DIR = "input_csv"           
OUTPUT_DIR = "dataset_audio"      
LOG_FILE = "master_log_t30.csv"   

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. FUNZIONE ESTRATTORE MATERIALI ---
def safe_float(val):
    try:
        v = float(str(val).replace(',', '.'))
        return 0.1 if np.isnan(v) else v
    except:
        return 0.1

def crea_materiale(row, prefix, surface_name):
    c_125 = safe_float(row[f'{prefix}_a125'])
    c_250 = safe_float(row[f'{prefix}_a250'])
    c_500 = safe_float(row[f'{prefix}_a500'])
    c_1k  = safe_float(row[f'{prefix}_a1000'])
    c_2k  = safe_float(row[f'{prefix}_a2000'])
    c_4k  = safe_float(row[f'{prefix}_a4000'])
    
    coeffs = [c_125, c_250, c_500, c_1k, c_2k, c_4k]
    coeffs_puliti = [float(x) for x in np.clip(coeffs, 0.02, 0.99)]
    
    # Sintassi esatta usata nel Modulo 3 (che non genera il bug)
    return pra.Material(
        energy_absorption={
            "description": f"M1_{surface_name}",
            "coeffs": coeffs_puliti,
            "center_freqs": [125, 250, 500, 1000, 2000, 4000]
        },
        scattering=0.15
    )

# --- 3. MOTORE PRINCIPALE ---
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
            room_type = row['room_type']
            
            try:
                # 1. Geometria
                L, W, H = float(row['L_m']), float(row['W_m']), float(row['H_m'])
                src_pos = [float(row['src_x']), float(row['src_y']), float(row['src_z'])]
                mic_pos = [float(row['mic_x']), float(row['mic_y']), float(row['mic_z'])]
                
                # 2. Assegnazione Materiali
                mats = {
                    "floor": crea_materiale(row, 'floor', 'floor'),
                    "ceiling": crea_materiale(row, 'ceiling', 'ceiling'),
                    "east": crea_materiale(row, 'wall_east', 'east'),
                    "west": crea_materiale(row, 'wall_west', 'west'),
                    "north": crea_materiale(row, 'wall_north', 'north'),
                    "south": crea_materiale(row, 'wall_south', 'south')
                }
                
                # 3 e 4. Creazione Stanza e Calcolo RIR (Con Jittering "Macro" per bypassare il bug PRA)
                max_retries = 5
                rir = None
                
                for attempt in range(max_retries):
                    try:
                        room = pra.ShoeBox(
                            [L, W, H], materials=mats, fs=44100, max_order=15, 
                            ray_tracing=True, air_absorption=True
                        )
                        room.set_ray_tracing(receiver_radius=0.1, n_rays=10000, energy_thres=1e-6)
                        room.add_source(src_pos)
                        
                        # Jittering MACRO: Se PRA va in crash, spostiamo il mic di +- 5 cm
                        current_mic_pos = mic_pos.copy()
                        if attempt > 0:
                            # 5 cm garantiscono uno shift di svariati sample audio (1 sample = 7.7mm)
                            current_mic_pos = [m + np.random.uniform(-0.05, 0.05) for m in mic_pos]
                            print(f"  ⚠️ Jittering (spostamento di ~5cm) applicato alla stanza {variant_id} (tentativo {attempt+1}/{max_retries})")
                            
                        mic_locs = np.array(current_mic_pos).reshape(3, 1)
                        mic_array = pra.MicrophoneArray(mic_locs, room.fs)
                        room.add_microphone_array(mic_array)
                        
                        room.compute_rir()
                        rir = room.rir[0][0]
                        break  # Uscita trionfale se non crasha!
                        
                    except ValueError as rir_err:
                        if "operands could not be broadcast" in str(rir_err):
                            if attempt == max_retries - 1:
                                print(f"  ❌ Fallimento definitivo per la stanza {variant_id}.")
                                rir = np.array([])
                        else:
                            raise rir_err
                
                # Evitiamo crash se la RIR è vuota o scartata
                if rir is None or len(rir) == 0 or np.max(np.abs(rir)) < 1e-15:
                    print(f"  ⚠️ RIR nulla o non calcolabile alla stanza ID {variant_id}")
                    continue
                    
                rir = rir / np.max(np.abs(rir))
                
                # 5. Estrazione T30 Multi-Banda
                t30_dict = extract_t30_per_band(rir, 44100)
                
                # 6. Salvataggio Audio
                nome_audio = f"{room_type}_{int(variant_id):03d}.wav"
                path_audio = os.path.join(out_sub_dir, nome_audio)
                sf.write(path_audio, rir, 44100)
                
                # 7. Aggiornamento Log
                log_entry = {
                    "file_audio": f"{nome_base}/{nome_audio}",
                    "room_type": room_type,
                    "volume_m3": row['V_m3'],
                    "t30_1000Hz": t30_dict.get('t30_1000'),
                    "t30_2000Hz": t30_dict.get('t30_2000'),
                    "t30_4000Hz": t30_dict.get('t30_4000'),
                    "t30_broadband": t30_dict.get('t30_broadband')
                }
                log_totale.append(log_entry)
                
                print(f"  ✅ Generata stanza {variant_id} (T30 Broadband: {t30_dict.get('t30_broadband')}s)")
                    
            except Exception as e:
                print(f"  ⚠️ Errore alla stanza ID {variant_id}: {e}")

        gc.collect() 
        print(f"--- Completato: {nome_base} ---\n")

    pd.DataFrame(log_totale).to_csv(LOG_FILE, index=False)
    print(f"🎉 TUTTO COMPLETATO! Generato il log master in '{LOG_FILE}'")

if __name__ == "__main__":
    main()