import numpy as np, pandas as pd

df  = pd.read_excel('input_csv/bagno_variants_150.xlsx')
log = pd.read_csv('master_log_t30.csv', sep=';', decimal=',')

freqs   = [125, 250, 500, 1000, 2000, 4000]
labels  = ['t30_125','t30_250','t30_500','t30_1k','t30_2k','t30_4k']
surfaces = ['floor','ceiling','wall_east','wall_west','wall_north','wall_south']

df['S_floor']  = df.L_m * df.W_m;  df['S_ceiling'] = df.L_m * df.W_m
df['S_east']   = df.W_m * df.H_m;  df['S_west']    = df.W_m * df.H_m
df['S_north']  = df.L_m * df.H_m;  df['S_south']   = df.L_m * df.H_m

area_map = {'floor':'S_floor','ceiling':'S_ceiling','wall_east':'S_east',
            'wall_west':'S_west','wall_north':'S_north','wall_south':'S_south'}

CLIP_MIN = 0.02

print(f"{'Banda':>8} | {'A_eff':>6} | {'Sabine':>7} | {'Sim':>7} | {'Ratio':>6} | {'Giudizio'}")
print("-" * 65)
for f, lbl in zip(freqs, labels):
    alpha_eff = {s: np.clip(df[f'{s}_a{f}'], CLIP_MIN, 0.99) for s in surfaces}
    A = sum(df[area_map[s]] * alpha_eff[s] for s in surfaces)
    T_sab = (0.161 * df.V_m3 / A).mean()
    T_sim = log[lbl].mean()
    ratio = T_sim / T_sab
    # in campo diffuso Sabine sottostima sempre un po': ratio 1.0-1.6 è fisiologico
    ok = "OK" if 0.9 <= ratio <= 1.8 else ("ALTO" if ratio > 1.8 else "BASSO")
    print(f"{f:>6} Hz | {A.mean():>6.2f} | {T_sab:>7.2f}s | {T_sim:>7.2f}s | {ratio:>6.2f}x | {ok}")

print()
print("--- Assorbimento medio per superficie a 125 Hz (con clip 0.05) ---")
for s in surfaces:
    raw  = df[f'{s}_a125'].mean()
    clip = np.clip(df[f'{s}_a125'], CLIP_MIN, 0.99).mean()
    changed = "← alzato" if clip > raw + 0.001 else ""
    print(f"  {s:<15} raw={raw:.3f}  clipped={clip:.3f}  {changed}")
