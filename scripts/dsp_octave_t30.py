"""
dsp_octave_t30.py
=================
Modulo DSP per l'estrazione del T30 su 6 bande d'ottava (125 Hz – 4 kHz)
da una Room Impulse Response (RIR) generata con pyroomacoustics.

Sostituisce l'approccio broadband `room.measure_rt60()[0][0]` con una
pipeline per-banda basata su scipy.signal, restituendo 7 valori per stanza:
    t30_125, t30_250, t30_500, t30_1k, t30_2k, t30_4k, t30_broadband

Progettato per essere integrato nello script universale esistente senza
modificare la logica di generazione PRA (ShoeBox, ray tracing, safe_float, ecc.)

Autore: AI-Driven XR Acoustic Simulator — Fase 1/3 Dataset
"""

import numpy as np
from math import gcd
from scipy.signal import butter, sosfilt, resample_poly
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Definizione delle bande d'ottava (Hz) — standard ISO 3382-1
# Formato: (banda_nominale, f_low, f_high)
# I limiti sono i punti a -3dB della banda d'ottava: f_center / sqrt(2) e f_center * sqrt(2)
# ---------------------------------------------------------------------------
OCTAVE_BANDS = [
    (125,   88.4,   176.8),
    (250,  176.8,   353.6),
    (500,  353.6,   707.1),
    (1000, 707.1,  1414.2),
    (2000, 1414.2, 2828.4),
    (4000, 2828.4, 5656.9),
]

# Label per le colonne CSV
BAND_LABELS = ["t30_125", "t30_250", "t30_500", "t30_1k", "t30_2k", "t30_4k"]

# ---------------------------------------------------------------------------
# Parametri filtro
# ---------------------------------------------------------------------------
FILTER_ORDER = 4        # Butterworth 4° ordine: buon compromesso ripple/roll-off
                        # Aumentare a 6 per roll-off più netto se le bande si sovrappongono
                        # Diminuire a 2 per RIR molto corte (evita instabilità su segnali brevi)

MIN_RIR_SAMPLES = 512   # Sotto questa soglia il filtro è inaffidabile
T30_FALLBACK    = np.nan  # Valore di fallback in caso di errore

# Sample rate usato internamente per il filtraggio per-banda.
# A 44100 Hz le frequenze normalizzate della banda 125 Hz (88/22050 = 0.004)
# sono troppo vicine allo zero: i coefficienti Butterworth perdono precisione
# e l'EDC di Schroeder risulta gonfiata (T30 fino a 4x Sabine).
# Ricampionando a 16000 Hz si ottiene 88/8000 = 0.011 → filtro stabile.
# Il limite superiore della banda 4 kHz è 5657 Hz < Nyquist(16000) = 8000 Hz ✓
FS_ANALYSIS = 16000


def _design_bandpass_filter(f_low: float, f_high: float, fs: float, order: int = FILTER_ORDER):
    """
    Progetta un filtro Butterworth passa-banda in formato Second-Order Sections (SOS).

    Usiamo SOS invece di (b, a) per evitare instabilità numerica sui filtri di
    ordine elevato — specialmente critico per bande basse (125 Hz) dove la
    frequenza normalizzata è molto piccola e i coefficienti (b, a) diventano
    mal condizionati.

    Parameters
    ----------
    f_low  : frequenza di taglio inferiore [Hz]
    f_high : frequenza di taglio superiore [Hz]
    fs     : sample rate [Hz]
    order  : ordine del filtro (default 4)

    Returns
    -------
    sos : array (n_sections, 6) — coefficienti SOS pronti per sosfilt()
    """
    nyq = fs / 2.0
    low  = f_low  / nyq
    high = f_high / nyq

    # Clamp per evitare valori fuori range [0, 1] dovuti a errori di arrotondamento
    low  = np.clip(low,  1e-6, 1 - 1e-6)
    high = np.clip(high, 1e-6, 1 - 1e-6)

    if low >= high:
        raise ValueError(f"Banda non valida: f_low={f_low} >= f_high={f_high} con fs={fs}")

    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sos


def _apply_filter(rir: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """
    Applica il filtro SOS alla RIR mono.

    sosfilt() elabora campione per campione evitando accumulo di errori
    floating-point sulle sezioni del secondo ordine.
    """
    return sosfilt(sos, rir)


def _schroeder_integral(rir_band: np.ndarray) -> np.ndarray:
    """
    Calcola l'integrale di Schroeder (backward integration) del segnale filtrato.

    L'integrale di Schroeder EDC (Energy Decay Curve) è la base per estrarre
    T30, T20, EDT — standard ISO 3382-1.

    EDC[n] = sum(rir_band[n:]^2) — calcolato con cumsum inverso per efficienza O(N).

    Returns
    -------
    edc_db : EDC normalizzata in dB (0 dB al picco)
    """
    energy = rir_band ** 2

    # Integrazione backward: equivale a sum(energy[n:]) per ogni n
    edc = np.cumsum(energy[::-1])[::-1]

    # Evita log(0): sostituisce zero con il minimo positivo rappresentabile
    edc = np.where(edc > 0, edc, np.finfo(float).tiny)

    # Normalizza in dB rispetto al picco
    edc_db = 10.0 * np.log10(edc / edc[0])
    return edc_db


def _extract_t30_from_edc(edc_db: np.ndarray, fs: float) -> float:
    """
    Estrae il T30 dall'EDC con regressione lineare tra -5 dB e -35 dB.

    Il T30 è il tempo che impiega la curva di decadimento a scendere di 30 dB
    nel range -5 / -35 dB, poi estrapolato a 60 dB (ISO 3382-1 §A.2).

    Questa funzione replica la logica interna di pyroomacoustics ma applicata
    al segnale per-banda, eliminando il bug di broadcast che affligge
    `room.measure_rt60()` su certe configurazioni.

    Parameters
    ----------
    edc_db : EDC normalizzata in dB, shape (N,)
    fs     : sample rate [Hz]

    Returns
    -------
    t30 : float [secondi], o np.nan se la regressione fallisce
    """
    time_axis = np.arange(len(edc_db)) / fs

    # Maschera: punti nell'intervallo [-5, -35] dB
    mask = (edc_db <= -5.0) & (edc_db >= -35.0)

    if mask.sum() < 10:
        # Troppo pochi punti per una regressione affidabile
        logger.debug("T30: meno di 10 campioni nell'intervallo [-5,-35] dB — NaN restituito")
        return T30_FALLBACK

    t_sel   = time_axis[mask]
    edc_sel = edc_db[mask]

    # Regressione lineare: edc ≈ slope * t + intercept
    coeffs = np.polyfit(t_sel, edc_sel, 1)   # [slope, intercept]
    slope, intercept = coeffs

    if slope >= 0:
        # EDC non decrescente: stanza non fisicamente realistica o RIR corrotta
        logger.warning("T30: slope EDC ≥ 0 (%.4f) — NaN restituito", slope)
        return T30_FALLBACK

    # Estrapolazione: quanto tempo per -60 dB?
    # t30 = (target - intercept) / slope, target = -60 dB
    # Per definizione ISO: t30 = 2 * (t_-35dB - t_-5dB)
    # Usiamo la regressione per robustezza al rumore
    t30 = (-60.0 - intercept) / slope

    if t30 <= 0 or t30 > 30.0:
        # Valori fisicamente impossibili (stanza reale: max ~10s, studio professionale)
        logger.warning("T30 fuori range fisico: %.3f s — NaN restituito", t30)
        return T30_FALLBACK

    return float(t30)


# ---------------------------------------------------------------------------
# API pubblica — questa è l'unica funzione da chiamare dallo script principale
# ---------------------------------------------------------------------------

def extract_t30_per_band(
    rir: np.ndarray,
    fs: float,
    include_broadband: bool = True
) -> dict:
    """
    Estrae il T30 per le 6 bande d'ottava standard (125 Hz – 4 kHz) da una RIR mono.

    Questa funzione sostituisce `room.measure_rt60()[0][0]` nello script universale.
    Mantiene la stessa robustezza `try/except` già implementata per il bug di broadcast.

    Parameters
    ----------
    rir : np.ndarray
        Room Impulse Response mono, shape (N,). Se la RIR è stereo o multi-canale,
        passare solo il primo canale: `room.rir[0][0]`.
    fs : float
        Sample rate in Hz (tipicamente 44100.0).
    include_broadband : bool
        Se True aggiunge "t30_broadband" calcolato sull'EDC senza filtraggio.
        Utile come feature diagnostica nel dataset. Default True.

    Returns
    -------
    dict con chiavi:
        "t30_125", "t30_250", "t30_500", "t30_1k", "t30_2k", "t30_4k"
        Opzionale: "t30_broadband"
        Tutti i valori sono float [secondi] o np.nan se il calcolo fallisce.

    Esempio di integrazione nello script universale
    -----------------------------------------------
        # Sostituire queste righe:
        #   t30_val = room.measure_rt60()[0][0]
        #   row["t30"] = t30_val

        # Con:
        rir_signal = room.rir[0][0]
        t30_dict   = extract_t30_per_band(rir_signal, fs=SAMPLE_RATE)
        row.update(t30_dict)
    """
    # Validazione input
    if rir is None or len(rir) == 0:
        logger.error("RIR vuota o None ricevuta")
        return _empty_result(include_broadband)

    # Assicura array 1D float64
    rir = np.asarray(rir, dtype=np.float64).squeeze()
    if rir.ndim != 1:
        logger.error("RIR deve essere mono (1D), shape ricevuta: %s", rir.shape)
        return _empty_result(include_broadband)

    if len(rir) < MIN_RIR_SAMPLES:
        logger.warning(
            "RIR troppo corta (%d campioni < %d minimo) — tutti NaN",
            len(rir), MIN_RIR_SAMPLES
        )
        return _empty_result(include_broadband)

    results = {}

    # --- Calcolo broadband sull'originale (fs originale, nessun filtro) ---
    if include_broadband:
        try:
            edc_bb = _schroeder_integral(rir)
            results["t30_broadband"] = _extract_t30_from_edc(edc_bb, fs)
        except Exception as exc:
            logger.warning("T30 broadband fallito: %s", exc)
            results["t30_broadband"] = T30_FALLBACK

    # --- Ricampionamento a FS_ANALYSIS prima del filtraggio per-banda ---
    # Alla frequenza di acquisizione (tipicamente 44100 Hz) la banda 125 Hz ha
    # frequenza normalizzata 88/22050 = 0.004, troppo vicina allo zero per il
    # Butterworth: i coefficienti perdono precisione e l'EDC risulta gonfiata.
    # A FS_ANALYSIS = 16000 Hz → 88/8000 = 0.011, intervallo stabile.
    fs_int = int(round(fs))
    if fs_int != FS_ANALYSIS:
        g   = gcd(FS_ANALYSIS, fs_int)
        rir_ds = resample_poly(rir, FS_ANALYSIS // g, fs_int // g)
        fs_ds  = float(FS_ANALYSIS)
    else:
        rir_ds = rir
        fs_ds  = fs

    # --- Calcolo per-banda sul segnale ricampionato ---
    for (f_nom, f_low, f_high), label in zip(OCTAVE_BANDS, BAND_LABELS):
        try:
            sos     = _design_bandpass_filter(f_low, f_high, fs_ds)
            rir_band = _apply_filter(rir_ds, sos)
            edc_db  = _schroeder_integral(rir_band)
            t30_val = _extract_t30_from_edc(edc_db, fs_ds)
            results[label] = t30_val
        except Exception as exc:
            logger.warning(
                "T30 per banda %d Hz fallito: %s", f_nom, exc
            )
            results[label] = T30_FALLBACK

    return results


def _empty_result(include_broadband: bool) -> dict:
    """Restituisce un dizionario di NaN per tutte le colonne — usato nei casi di errore."""
    result = {label: T30_FALLBACK for label in BAND_LABELS}
    if include_broadband:
        result["t30_broadband"] = T30_FALLBACK
    return result


# ---------------------------------------------------------------------------
# Patch per lo script universale — snippet di integrazione commentato
# ---------------------------------------------------------------------------
#
# PRIMA (approccio broadband):
# ----------------------------
#   try:
#       t30_val = room.measure_rt60()[0][0]
#   except ValueError as e:
#       if "operands could not be broadcast" in str(e):
#           t30_val = np.nan
#       else:
#           raise
#   row["t30"] = t30_val
#
#
# DOPO (approccio per-banda, drop-in replacement):
# ------------------------------------------------
#   from dsp_octave_t30 import extract_t30_per_band
#
#   rir_signal = room.rir[0][0]        # RIR mono dal primo mic
#   t30_dict   = extract_t30_per_band(rir_signal, fs=SAMPLE_RATE)
#   row.update(t30_dict)
#   # row ora contiene: t30_125, t30_250, t30_500, t30_1k, t30_2k, t30_4k, t30_broadband
#
#
# Aggiornare anche le colonne del DataFrame / CSV:
# ------------------------------------------------
#   BAND_COLS = ["t30_125", "t30_250", "t30_500", "t30_1k", "t30_2k", "t30_4k", "t30_broadband"]
#   # Sostituire "t30" con BAND_COLS nell'inizializzazione del DataFrame
#   # e nell'intestazione del master_log_t30.csv
#
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test locale — eseguire con: python dsp_octave_t30.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pyroomacoustics as pra

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")

    print("=" * 60)
    print("Test: estrazione T30 per-banda su stanza ShoeBox sintetica")
    print("=" * 60)

    # Stanza di test: ufficio medio con assorbimento realistico
    FS          = 44100
    ROOM_DIM    = [8.0, 6.0, 3.0]   # L x W x H [m]
    SRC_POS     = [2.0, 2.0, 1.5]
    MIC_POS     = [6.0, 4.0, 1.2]

    # Coefficienti di assorbimento per banda [125, 250, 500, 1k, 2k, 4k]
    # Esempio: pavimento in parquet, pareti intonaco, soffitto controsoffitto
    materials = pra.Material(
        energy_absorption={
            "coeffs": [0.05, 0.07, 0.10, 0.12, 0.13, 0.14],
            "center_freqs": [125, 250, 500, 1000, 2000, 4000]
        }
    )

    room = pra.ShoeBox(
        ROOM_DIM,
        fs=FS,
        materials=materials,
        max_order=15,
        ray_tracing=True,
        air_absorption=True,
    )
    room.add_source(SRC_POS)
    room.add_microphone(MIC_POS)
    # compute_rir() genera solo la RIR senza richiedere un segnale sorgente,
    # coerente con lo script principale genera_dataset_xr.py.
    # simulate() richiederebbe add_source(pos, signal=...) e causerebbe
    # TypeError: object of type 'NoneType' has no len()
    room.compute_rir()

    rir_signal = room.rir[0][0]
    print(f"\nRIR generata: {len(rir_signal)} campioni ({len(rir_signal)/FS:.3f} s)\n")

    t30_results = extract_t30_per_band(rir_signal, fs=float(FS))

    print(f"{'Banda':<20} {'T30 [s]':>10}")
    print("-" * 32)
    for key, val in t30_results.items():
        label = key.replace("t30_", "").upper()
        if np.isnan(val):
            print(f"{label:<20} {'NaN':>10}  ← banda problematica")
        else:
            print(f"{label:<20} {val:>10.4f}")

    print("\nCSV columns da aggiungere al master_log_t30.csv:")
    print(", ".join(t30_results.keys()))