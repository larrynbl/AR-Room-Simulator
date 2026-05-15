import os
import glob
import numpy as np
import pandas as pd
import pyroomacoustics as pra
import soundfile as sf
import gc 

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

def crea_materiale(row, prefix):
    c_125 = safe_float(row[f'{prefix}_a125'])
    c_250 = safe_float(row[f'{prefix}_a250'])
    c_500 = safe_float(row[f'{prefix}_a500'])
    c_1k  = safe_float(row[f'{prefix}_a1000'])
    c_2k  = safe_float(row[f'{prefix}_a2000'])
    c_4k  = safe_float(row[f'{prefix}_a4000'])
    
    coeffs = [c_125, c_250, c_500, c_1k, c_2k, c_4k]
    coeffs_puliti = [float(x) for x in np.clip(coeffs, 0.01, 0.99)]
    
    return {
        "description": f"Custom_{prefix}",
        "coeffs": coeffs_puliti,
        "center_freqs": [125, 250, 500, 1000, 2000, 4000]
    }

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
                
                # 2. Assegnazione
                mats = pra.make_materials(
                    floor=crea_materiale(row, 'floor'),
                    ceiling=crea_materiale(row, 'ceiling'),
                    east=crea_materiale(row, 'wall_east'),
                    west=crea_materiale(row, 'wall_west'),
                    north=crea_materiale(row, 'wall_north'),
                    south=crea_materiale(row, 'wall_south')
                )
                
                # 3. Creazione Stanza
                room = pra.ShoeBox(
                    [L, W, H], materials=mats, fs=44100, max_order=15, 
                    ray_tracing=True, air_absorption=True
                )
                room.set_ray_tracing(receiver_radius=0.1, n_rays=10000, energy_thres=1e-5)
                room.add_source(src_pos)
                room.add_microphone(mic_pos)
                
                # 4. Calcolo RIR e T30
                room.compute_rir()
                rir = room.rir[0][0]
                rir = rir / np.max(np.abs(rir)) 
                
                t30_globale = float(room.measure_rt60()[0][0])
                
                # 5. Salvataggio
                nome_audio = f"{room_type}_{int(variant_id):03d}.wav"
                path_audio = os.path.join(out_sub_dir, nome_audio)
                sf.write(path_audio, rir, 44100)
                
                log_totale.append({
                    "file_audio": f"{nome_base}/{nome_audio}",
                    "room_type": room_type,
                    "volume_m3": row['V_m3'],
                    "t30_target_globale": round(t30_globale, 3)
                })
                
                print(f"  ✅ Generata stanza {variant_id} (T30: {t30_globale:.2f}s)")
                    
            except Exception as e:
                print(f"  ⚠️ Errore alla stanza ID {variant_id}: {e}")

        gc.collect() 
        print(f"--- Completato: {nome_base} ---\n")

    pd.DataFrame(log_totale).to_csv(LOG_FILE, index=False)
    print(f"🎉 TUTTO COMPLETATO! Generato il log master in '{LOG_FILE}'")

if __name__ == "__main__":
    main()