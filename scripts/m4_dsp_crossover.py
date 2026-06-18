"""
=============================================================================
MODULO 4 — DSP Crossover (HybridCrossover)
=============================================================================
Simulatore Acustico Ibrido XR — HPC-Ready Pipeline

Responsabilità:
  - Resampling della RIR_LF (campionamento k-Wave, dipendente da dx/CFL)
    alla frequenza di campionamento target di PRA (default 44100 Hz).
  - Allineamento temporale al singolo sample dei suoni diretti delle due
    RIR (argmax del picco principale), tramite zero-padding di testa.
  - Filtraggio Linkwitz-Riley di ordine 4 (concettuale: 2x Butterworth
    cascade applicate in zero-phase tramite sosfiltfilt).
  - Level matching basato sull'energia RMS in banda ristretta attorno
    alla frequenza di crossover f_s.
  - Somma e normalizzazione finale → RIR_Hybrid.

Note teoriche
-------------
Il filtro Linkwitz-Riley classico (LR4) si ottiene cascando due
Butterworth di ordine 2: la risposta complessiva è -24 dB/oct con
fase concorde tra LP e HP a f_s (entrambi -6 dB a f_s, somma piatta).

Qui usiamo Butterworth ordine 4 + sosfiltfilt (zero-phase, forward-backward).
sosfiltfilt raddoppia l'ordine effettivo (ord 4 -> 8) ma annulla la fase,
quindi la somma LP + HP non genera notch dovuti a sfasamenti.
È la prassi standard per le RIR ibride in letteratura.

Input  : RIR_LF (k-Wave, fs_lf), RIR_HF (PRA, fs_hf), f_s
Output : RIR_Hybrid (np.ndarray 1D float64, fs_hf, picco unitario)

Autore  : Senior Audio DSP / Acoustic Simulation Engineer
Versione: 1.0.0
Python  : >= 3.10
Dipendenze: numpy, scipy
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fractions import Fraction
from typing import Final, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, resample_poly, sosfiltfilt

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
LR4_BUTTER_ORDER:    Final[int]   = 4        # Butterworth + sosfiltfilt -> ord effettivo 8 (LR4-like)
LEVEL_MATCH_BW_OCT:  Final[float] = 1.0 / 3  # banda di analisi ±1/3 ottava attorno a f_s
RESAMPLE_RATIO_TOL:  Final[float] = 1e-6     # tolleranza per skip resampling
MIN_GAIN_DB:         Final[float] = -30.0    # clamp inferiore del gain di matching [dB]
MAX_GAIN_DB:         Final[float] = +30.0    # clamp superiore del gain di matching [dB]


# ---------------------------------------------------------------------------
# Eccezione custom
# ---------------------------------------------------------------------------

class CrossoverError(RuntimeError):
    """
    Sollevata quando una qualsiasi fase del crossover fallisce
    (resampling impossibile, RIR vuote, f_s fuori range, ecc.).
    """
    pass


# ---------------------------------------------------------------------------
# Dataclass diagnostica
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CrossoverDiagnostics:
    """
    Riassunto delle metriche di processing del crossover.
    Utile per logging dataset-wide e debug per-stanza.
    """
    fs_target:           int
    f_crossover:         float
    rir_lf_len_original: int
    rir_hf_len_original: int
    rir_lf_len_resampled: int
    direct_peak_lf:      int    # sample index dopo allineamento
    direct_peak_hf:      int
    shift_lf_samples:    int    # zero-padding applicato in testa a LF
    shift_hf_samples:    int    # zero-padding applicato in testa a HF
    rms_lf_in_band:      float
    rms_hf_in_band:      float
    matching_gain_db:    float
    final_len_samples:   int


# ---------------------------------------------------------------------------
# HybridCrossover
# ---------------------------------------------------------------------------

class HybridCrossover:
    """
    Esegue la fusione DSP delle due RIR (LF da k-Wave, HF da PRA) in
    una RIR ibrida full-band.

    Pipeline (metodo merge)
    -----------------------
        1. resample_lf      — RIR_LF -> fs_hf (resample_poly)
        2. align_temporal   — argmax-based alignment con zero-padding di testa
        3. lr4_filter       — LP(fs_lf) su RIR_LF, HP(fs_lf) su RIR_HF
        4. level_match      — gain RMS in banda ±1/3 ottava attorno a f_s
        5. sum_and_normalize — somma e normalizzazione picco
    """

    def __init__(self, butter_order: int = LR4_BUTTER_ORDER) -> None:
        """
        Parameters
        ----------
        butter_order : int
            Ordine del filtro Butterworth. Default = 4.
            sosfiltfilt raddoppia l'ordine effettivo (≈ LR di ordine 2*order).
        """
        if butter_order < 2:
            raise ValueError(
                f"butter_order >= 2 richiesto per LR-like, ricevuto {butter_order}."
            )
        self.butter_order = butter_order
        logger.info(
            "HybridCrossover inizializzato: butter_order=%d "
            "(ordine effettivo zero-phase ≈ %d)",
            butter_order, 2 * butter_order,
        )

    # ------------------------------------------------------------------ #
    #  1. Resampling                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resample_to_target(
        x:         NDArray[np.float64],
        fs_in:     float,
        fs_target: int,
    ) -> NDArray[np.float64]:
        """
        Resample x da fs_in a fs_target usando scipy.signal.resample_poly.

        resample_poly è preferito a resample (FFT) perché:
          - Funziona su segnali lunghi senza moltiplicare la RAM
          - Applica un filtro polifase Kaiser anti-aliasing implicito
          - Preserva meglio i transienti (importanti per RIR)

        La conversione fs_in -> fs_target viene espressa come frazione
        razionale up/down tramite fractions.Fraction(...).limit_denominator(1000),
        che è la prassi per fs irrazionali come 1/dt di k-Wave.

        Parameters
        ----------
        x : NDArray
            Segnale di input 1D.
        fs_in : float
            Sample rate di x [Hz].
        fs_target : int
            Sample rate di output [Hz].

        Returns
        -------
        NDArray
            x resampled a fs_target.
        """
        if fs_in <= 0:
            raise CrossoverError(f"fs_in = {fs_in} <= 0: invalido.")

        if abs(fs_in - fs_target) / fs_target < RESAMPLE_RATIO_TOL:
            logger.info(
                "Resampling skippato: fs_in (%.2f Hz) ≈ fs_target (%d Hz).",
                fs_in, fs_target,
            )
            return x.copy()

        # Costruisci ratio razionale up/down
        ratio = Fraction(fs_target / fs_in).limit_denominator(10000)
        up, down = ratio.numerator, ratio.denominator

        logger.info(
            "Resampling: fs_in=%.2f Hz -> fs_target=%d Hz | up=%d, down=%d | "
            "ratio approssimato=%.6f (vero=%.6f)",
            fs_in, fs_target, up, down,
            up / down, fs_target / fs_in,
        )

        x_resampled = resample_poly(x, up=up, down=down)

        logger.info(
            "Resampling: %d -> %d samples (atteso ~%d)",
            len(x), len(x_resampled),
            int(round(len(x) * fs_target / fs_in)),
        )
        return x_resampled.astype(np.float64)

    # ------------------------------------------------------------------ #
    #  2. Allineamento temporale                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _align_temporal(
        rir_lf: NDArray[np.float64],
        rir_hf: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], int, int]:
        """
        Allinea le due RIR sui rispettivi picchi diretti (argmax del valore
        assoluto) tramite zero-padding di testa della RIR che "arriva prima".

        Logica
        ------
            peak_lf = argmax(|rir_lf|)
            peak_hf = argmax(|rir_hf|)
            target = max(peak_lf, peak_hf)
            shift_lf = target - peak_lf
            shift_hf = target - peak_hf
            -> zero-pad in testa di shift_lf samples su rir_lf
            -> zero-pad in testa di shift_hf samples su rir_hf

        Successivamente entrambe le RIR sono estese in coda allo stesso
        max len con zero-padding, così la successiva somma è elementwise.

        Returns
        -------
        Tuple[NDArray, NDArray, int, int]
            (rir_lf_aligned, rir_hf_aligned, shift_lf, shift_hf)
        """
        if len(rir_lf) == 0 or len(rir_hf) == 0:
            raise CrossoverError(
                f"RIR vuote in input: len(rir_lf)={len(rir_lf)}, "
                f"len(rir_hf)={len(rir_hf)}."
            )

        peak_lf = int(np.argmax(np.abs(rir_lf)))
        peak_hf = int(np.argmax(np.abs(rir_hf)))
        target  = max(peak_lf, peak_hf)

        shift_lf = target - peak_lf
        shift_hf = target - peak_hf

        rir_lf_padded = np.concatenate([
            np.zeros(shift_lf, dtype=np.float64),
            rir_lf,
        ])
        rir_hf_padded = np.concatenate([
            np.zeros(shift_hf, dtype=np.float64),
            rir_hf,
        ])

        # Estendi in coda alla stessa lunghezza
        max_len = max(len(rir_lf_padded), len(rir_hf_padded))
        rir_lf_aligned = np.pad(rir_lf_padded, (0, max_len - len(rir_lf_padded)))
        rir_hf_aligned = np.pad(rir_hf_padded, (0, max_len - len(rir_hf_padded)))

        logger.info(
            "Allineamento temporale: peak_lf=%d, peak_hf=%d, target=%d | "
            "shift_lf=%d sample, shift_hf=%d sample | "
            "len finale=%d",
            peak_lf, peak_hf, target,
            shift_lf, shift_hf, max_len,
        )

        # Sanity check post-allineamento
        new_peak_lf = int(np.argmax(np.abs(rir_lf_aligned)))
        new_peak_hf = int(np.argmax(np.abs(rir_hf_aligned)))
        if new_peak_lf != new_peak_hf:
            logger.warning(
                "Post-allineamento: argmax LF=%d != argmax HF=%d. "
                "Possibile presenza di picchi spurii più alti del diretto.",
                new_peak_lf, new_peak_hf,
            )
        else:
            logger.info("Allineamento verificato: entrambi i picchi al sample %d.", new_peak_lf)

        return rir_lf_aligned, rir_hf_aligned, shift_lf, shift_hf

    # ------------------------------------------------------------------ #
    #  3. Filtraggio Linkwitz-Riley (concettuale)                          #
    # ------------------------------------------------------------------ #

    def _lr4_lowpass(
        self,
        x:     NDArray[np.float64],
        fs:    int,
        f_cut: float,
    ) -> NDArray[np.float64]:
        """
        Passa-basso Butterworth zero-phase a f_cut.
        Ordine effettivo dopo sosfiltfilt: 2 * butter_order.
        """
        nyq = fs / 2.0
        if not (0 < f_cut < nyq):
            raise CrossoverError(
                f"f_cut={f_cut} Hz fuori dal range (0, Nyquist={nyq})."
            )
        sos = butter(self.butter_order, f_cut / nyq, btype="low", output="sos")
        y = sosfiltfilt(sos, x)
        logger.debug(
            "LP applicato: ord=%d, f_cut=%.2f Hz, |y|_max=%.4e",
            self.butter_order, f_cut, float(np.max(np.abs(y))),
        )
        return y

    def _lr4_highpass(
        self,
        x:     NDArray[np.float64],
        fs:    int,
        f_cut: float,
    ) -> NDArray[np.float64]:
        """
        Passa-alto Butterworth zero-phase a f_cut.
        """
        nyq = fs / 2.0
        if not (0 < f_cut < nyq):
            raise CrossoverError(
                f"f_cut={f_cut} Hz fuori dal range (0, Nyquist={nyq})."
            )
        sos = butter(self.butter_order, f_cut / nyq, btype="high", output="sos")
        y = sosfiltfilt(sos, x)
        logger.debug(
            "HP applicato: ord=%d, f_cut=%.2f Hz, |y|_max=%.4e",
            self.butter_order, f_cut, float(np.max(np.abs(y))),
        )
        return y

    # ------------------------------------------------------------------ #
    #  4. Level matching                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rms_in_band(
        x:    NDArray[np.float64],
        fs:   int,
        f_lo: float,
        f_hi: float,
    ) -> float:
        """
        Calcola l'RMS di x filtrato passa-banda [f_lo, f_hi] (zero-phase
        Butterworth ord 4). Usato per il level matching.

        Restituisce 0 se x è nullo o se f_lo >= f_hi.
        """
        if np.allclose(x, 0):
            return 0.0
        if f_lo >= f_hi:
            return 0.0

        nyq = fs / 2.0
        f_lo_safe = max(f_lo, 0.5)              # evita DC
        f_hi_safe = min(f_hi, nyq * 0.99)       # evita Nyquist

        sos = butter(
            4,
            [f_lo_safe / nyq, f_hi_safe / nyq],
            btype="band",
            output="sos",
        )
        x_band = sosfiltfilt(sos, x)
        return float(np.sqrt(np.mean(x_band ** 2)))

    def _level_match(
        self,
        rir_lf_filt: NDArray[np.float64],
        rir_hf_filt: NDArray[np.float64],
        fs:          int,
        f_s:         float,
    ) -> Tuple[NDArray[np.float64], float, float, float]:
        """
        Calcola il gain di matching che porta la RMS in banda di rir_hf_filt
        a coincidere con quella di rir_lf_filt nella banda [f_s/2^(1/3), f_s*2^(1/3)]
        (±1/3 di ottava attorno a f_s).

        Strategia
        ---------
        Si misura la RMS delle due RIR (già filtrate LP/HP) nella banda
        di transizione attorno a f_s, dove entrambe contribuiscono ~6 dB
        (per LR4) al crossover. Il rapporto RMS_LF/RMS_HF è il gain da
        applicare a rir_hf per pareggiare l'energia.

        Il gain viene clampato in [MIN_GAIN_DB, MAX_GAIN_DB] per evitare
        amplificazioni patologiche se una delle due RIR è quasi nulla
        nella banda (es. simulazione k-Wave divergente).

        Returns
        -------
        Tuple[NDArray, float, float, float]
            (rir_hf_matched, rms_lf, rms_hf, gain_db)
        """
        # Banda di analisi: ±1/3 ottava attorno a f_s
        oct_factor = 2 ** LEVEL_MATCH_BW_OCT     # ≈ 1.26
        f_lo = f_s / oct_factor
        f_hi = f_s * oct_factor

        rms_lf = self._rms_in_band(rir_lf_filt, fs, f_lo, f_hi)
        rms_hf = self._rms_in_band(rir_hf_filt, fs, f_lo, f_hi)

        if rms_hf < 1e-15:
            logger.warning(
                "Level matching: rms_hf ~ 0 (%.2e). Nessun matching applicato.",
                rms_hf,
            )
            return rir_hf_filt.copy(), rms_lf, rms_hf, 0.0

        if rms_lf < 1e-15:
            logger.warning(
                "Level matching: rms_lf ~ 0 (%.2e). Nessun matching applicato.",
                rms_lf,
            )
            return rir_hf_filt.copy(), rms_lf, rms_hf, 0.0

        gain_linear = rms_lf / rms_hf
        gain_db_raw = 20.0 * np.log10(gain_linear)
        gain_db     = float(np.clip(gain_db_raw, MIN_GAIN_DB, MAX_GAIN_DB))

        if gain_db != gain_db_raw:
            logger.warning(
                "Level matching: gain raw=%.2f dB clampato a %.2f dB "
                "([%g, %g] dB).",
                gain_db_raw, gain_db, MIN_GAIN_DB, MAX_GAIN_DB,
            )

        gain_linear_clamped = 10 ** (gain_db / 20.0)
        rir_hf_matched = rir_hf_filt * gain_linear_clamped

        logger.info(
            "Level matching: banda [%.1f, %.1f] Hz | "
            "RMS_LF=%.4e, RMS_HF=%.4e | gain=%.2f dB",
            f_lo, f_hi, rms_lf, rms_hf, gain_db,
        )
        return rir_hf_matched, rms_lf, rms_hf, gain_db

    # ------------------------------------------------------------------ #
    #  Entry point pubblico                                                #
    # ------------------------------------------------------------------ #

    def merge(
        self,
        rir_lf: NDArray[np.float64],
        fs_lf:  float,
        rir_hf: NDArray[np.float64],
        fs_hf:  float,
        f_s:    float,
        return_diagnostics: bool = False,
    ) -> NDArray[np.float64] | Tuple[NDArray[np.float64], CrossoverDiagnostics]:
        """
        Pipeline completa di fusione.

        Parameters
        ----------
        rir_lf : NDArray[float64]
            RIR a bassa frequenza (output Modulo 2 — k-Wave).
        fs_lf : float
            Sample rate di rir_lf [Hz] (= 1/dt di k-Wave, tipicamente
            non-standard).
        rir_hf : NDArray[float64]
            RIR ad alta frequenza (output Modulo 3 — PRA).
        fs_hf : float
            Sample rate di rir_hf [Hz] (tipicamente 44100).
        f_s : float
            Frequenza di crossover [Hz] (= Schroeder dal Modulo 1).
        return_diagnostics : bool
            Se True, restituisce anche un CrossoverDiagnostics.

        Returns
        -------
        NDArray[float64]   (o tupla con diagnostics)
            RIR_Hybrid full-band, sample rate = fs_hf, normalizzata a
            picco unitario.

        Raises
        ------
        CrossoverError
            Se input invalidi (RIR vuote, f_s fuori range, ecc.).
        """
        logger.info("=" * 60)
        logger.info("HybridCrossover.merge — START")
        logger.info(
            "  rir_lf: %d sample @ %.2f Hz | rir_hf: %d sample @ %.2f Hz | "
            "f_crossover=%.2f Hz",
            len(rir_lf), fs_lf, len(rir_hf), fs_hf, f_s,
        )
        logger.info("=" * 60)

        # -------- Validazione input --------
        if not isinstance(rir_lf, np.ndarray) or rir_lf.ndim != 1:
            raise CrossoverError(
                f"rir_lf deve essere np.ndarray 1D, ricevuto {type(rir_lf)} "
                f"{getattr(rir_lf, 'shape', '?')}"
            )
        if not isinstance(rir_hf, np.ndarray) or rir_hf.ndim != 1:
            raise CrossoverError(
                f"rir_hf deve essere np.ndarray 1D, ricevuto {type(rir_hf)} "
                f"{getattr(rir_hf, 'shape', '?')}"
            )

        fs_target = int(round(fs_hf))
        nyq = fs_target / 2.0
        if not (0 < f_s < nyq):
            raise CrossoverError(
                f"f_s={f_s} Hz fuori range (0, Nyquist={nyq})."
            )

        len_lf_orig = len(rir_lf)
        len_hf_orig = len(rir_hf)

        # -------- Step 1: Resampling LF -> fs_hf --------
        rir_lf_rs = self._resample_to_target(
            x=rir_lf.astype(np.float64),
            fs_in=fs_lf,
            fs_target=fs_target,
        )
        len_lf_rs = len(rir_lf_rs)
        rir_hf_f64 = rir_hf.astype(np.float64)

        # -------- Step 2: Allineamento temporale --------
        rir_lf_aligned, rir_hf_aligned, shift_lf, shift_hf = self._align_temporal(
            rir_lf_rs, rir_hf_f64
        )
        peak_lf_post = int(np.argmax(np.abs(rir_lf_aligned)))
        peak_hf_post = int(np.argmax(np.abs(rir_hf_aligned)))

        # -------- Step 3: Filtraggio LR4-like --------
        logger.info("Filtraggio LR4-like @ f_s = %.2f Hz", f_s)
        rir_lf_filt = self._lr4_lowpass (rir_lf_aligned, fs_target, f_s)
        rir_hf_filt = self._lr4_highpass(rir_hf_aligned, fs_target, f_s)

        # -------- Step 4: Level matching --------
        rir_hf_matched, rms_lf, rms_hf, gain_db = self._level_match(
            rir_lf_filt, rir_hf_filt, fs_target, f_s
        )

        # -------- Step 5: Somma e normalizzazione --------
        rir_hybrid = rir_lf_filt + rir_hf_matched
        peak = float(np.max(np.abs(rir_hybrid)))
        if peak < 1e-15:
            logger.warning(
                "RIR_Hybrid picco quasi nullo (%.2e). "
                "Le due RIR potrebbero cancellarsi reciprocamente.",
                peak,
            )
        else:
            rir_hybrid = rir_hybrid / peak

        logger.info(
            "HybridCrossover.merge DONE — RIR_Hybrid: %d sample @ %d Hz | "
            "peak originale=%.4e | gain matching=%+.2f dB",
            len(rir_hybrid), fs_target, peak, gain_db,
        )
        logger.info("=" * 60)

        if return_diagnostics:
            diag = CrossoverDiagnostics(
                fs_target=fs_target,
                f_crossover=f_s,
                rir_lf_len_original=len_lf_orig,
                rir_hf_len_original=len_hf_orig,
                rir_lf_len_resampled=len_lf_rs,
                direct_peak_lf=peak_lf_post,
                direct_peak_hf=peak_hf_post,
                shift_lf_samples=shift_lf,
                shift_hf_samples=shift_hf,
                rms_lf_in_band=rms_lf,
                rms_hf_in_band=rms_hf,
                matching_gain_db=gain_db,
                final_len_samples=len(rir_hybrid),
            )
            return rir_hybrid, diag

        return rir_hybrid


# ---------------------------------------------------------------------------
# Self-test — usa RIR_LF sintetica al posto di k-Wave
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n" + "=" * 60)
    print("MODULO 4 — Self Test (RIR_LF sintetica)")
    print("=" * 60)

    # ---- Parametri di test ----
    fs_hf = 44100
    fs_lf = 2765.48          # tipico output k-Wave (vedi Modulo 2 self test)
    f_s   = 187.52           # f_schroeder dal Modulo 1 (stanza 6x4x3)

    # ---- Fake RIR_LF: rumore bianco filtrato passa-basso + impulso ----
    #     Simula un output realistico di k-Wave: picco diretto + coda esponenziale
    rng = np.random.default_rng(42)
    n_lf = int(fs_lf * 1.0)  # 1 secondo
    t_lf = np.arange(n_lf) / fs_lf

    # Impulso a 5 ms + rumore con decadimento exp (T60 ~ 0.6 s)
    fake_lf = np.zeros(n_lf, dtype=np.float64)
    direct_idx_lf = int(0.005 * fs_lf)
    fake_lf[direct_idx_lf] = 1.0   # picco diretto
    # Coda: rumore filtrato LP * envelope esponenziale
    noise = rng.standard_normal(n_lf)
    sos_lp = butter(4, (f_s * 0.9) / (fs_lf / 2), btype="low", output="sos")
    noise_lp = sosfiltfilt(sos_lp, noise)
    envelope = np.exp(-t_lf * np.log(1000) / 0.6)   # decadimento per T60 ~ 0.6 s
    fake_lf += 0.5 * noise_lp * envelope
    fake_lf /= np.max(np.abs(fake_lf))

    # ---- Fake RIR_HF: simile ma a fs_hf, con picco a tempo diverso (test align) ----
    n_hf = int(fs_hf * 0.8)
    t_hf = np.arange(n_hf) / fs_hf
    fake_hf = np.zeros(n_hf, dtype=np.float64)
    direct_idx_hf = int(0.012 * fs_hf)   # picco diretto a 12 ms (≠ 5 ms del LF)
    fake_hf[direct_idx_hf] = 1.0
    noise_hf = rng.standard_normal(n_hf)
    sos_hp = butter(4, (f_s * 1.1) / (fs_hf / 2), btype="high", output="sos")
    noise_hp = sosfiltfilt(sos_hp, noise_hf)
    envelope_hf = np.exp(-t_hf * np.log(1000) / 0.5)
    fake_hf += 0.3 * noise_hp * envelope_hf
    fake_hf /= np.max(np.abs(fake_hf))

    print(f"\nInput fake_LF: {len(fake_lf)} sample @ {fs_lf:.2f} Hz | "
          f"direct peak @ sample {int(np.argmax(np.abs(fake_lf)))} "
          f"({int(np.argmax(np.abs(fake_lf)))/fs_lf*1000:.2f} ms)")
    print(f"Input fake_HF: {len(fake_hf)} sample @ {fs_hf} Hz | "
          f"direct peak @ sample {int(np.argmax(np.abs(fake_hf)))} "
          f"({int(np.argmax(np.abs(fake_hf)))/fs_hf*1000:.2f} ms)")
    print(f"f_crossover:   {f_s:.2f} Hz")

    # ---- Esegui crossover ----
    crossover = HybridCrossover()
    rir_hybrid, diag = crossover.merge(
        rir_lf=fake_lf,
        fs_lf=fs_lf,
        rir_hf=fake_hf,
        fs_hf=fs_hf,
        f_s=f_s,
        return_diagnostics=True,
    )

    # ---- Riassunto ----
    print(f"\nDiagnostica crossover:")
    print(f"  fs_target            = {diag.fs_target} Hz")
    print(f"  f_crossover          = {diag.f_crossover:.2f} Hz")
    print(f"  rir_lf_len_original  = {diag.rir_lf_len_original}")
    print(f"  rir_lf_len_resampled = {diag.rir_lf_len_resampled}")
    print(f"  rir_hf_len_original  = {diag.rir_hf_len_original}")
    print(f"  direct_peak_lf       = sample {diag.direct_peak_lf} "
          f"({diag.direct_peak_lf/fs_hf*1000:.2f} ms)")
    print(f"  direct_peak_hf       = sample {diag.direct_peak_hf} "
          f"({diag.direct_peak_hf/fs_hf*1000:.2f} ms)")
    print(f"  shift_lf_samples     = {diag.shift_lf_samples}")
    print(f"  shift_hf_samples     = {diag.shift_hf_samples}")
    print(f"  rms_lf_in_band       = {diag.rms_lf_in_band:.4e}")
    print(f"  rms_hf_in_band       = {diag.rms_hf_in_band:.4e}")
    print(f"  matching_gain_db     = {diag.matching_gain_db:+.2f} dB")
    print(f"  final_len_samples    = {diag.final_len_samples}")

    print(f"\nRIR_Hybrid: {len(rir_hybrid)} sample @ {fs_hf} Hz | "
          f"peak={float(np.max(np.abs(rir_hybrid))):.4f} | "
          f"argmax @ sample {int(np.argmax(np.abs(rir_hybrid)))} "
          f"({int(np.argmax(np.abs(rir_hybrid)))/fs_hf*1000:.2f} ms)")

    # ---- Verifica spettrale: somma piatta attorno a f_s? ----
    # Calcola la PSD della RIR ibrida e verifica che intorno a f_s non ci sia
    # un notch eccessivo (sarebbe segno di sfasamento residuo tra LP e HP)
    from scipy.signal import welch
    f_psd, psd = welch(rir_hybrid, fs=fs_hf, nperseg=2048)
    # Media PSD in ±1/2 ottava attorno a f_s
    band_lo = f_s / np.sqrt(2)
    band_hi = f_s * np.sqrt(2)
    psd_in_band = float(np.mean(psd[(f_psd >= band_lo) & (f_psd <= band_hi)]))
    psd_global = float(np.mean(psd[f_psd > 0]))
    notch_db = 10 * np.log10(psd_in_band / (psd_global + 1e-15))
    print(f"\nVerifica spettrale (notch attorno a f_s):")
    print(f"  PSD nella banda [{band_lo:.0f}, {band_hi:.0f}] Hz vs globale: "
          f"{notch_db:+.2f} dB")
    print(f"  (attesi valori prossimi a 0 dB; un notch < -6 dB indicherebbe "
          f"problemi di fase)")

    print("\n" + "=" * 60)
    print("Self test Modulo 4 completato.")
    print("=" * 60)