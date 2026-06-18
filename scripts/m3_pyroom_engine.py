"""
=============================================================================
MODULO 3 — Pyroomacoustics Wrapper (High-Frequency RIR)
=============================================================================
Simulatore Acustico Ibrido XR — HPC-Ready Pipeline

Responsabilità:
  - Wrapping della logica geometrica (Image Source Method + Ray Tracing)
    del simulatore pyroomacoustics dentro un'interfaccia OOP coerente con
    il resto della pipeline.
  - Costruzione dei materiali PRA a partire dal RoomAcousticProfile del
    Modulo 1, con clipping di sicurezza dei coefficienti alpha [0.02, 0.99].
  - Esecuzione di una pra.ShoeBox in modalità ibrida ISM (max_order=15)
    + Ray Tracing (10000 raggi), che è la combinazione validata sul vecchio
    script di produzione.
  - Gestione del noto bug di broadcast PRA "operands could not be broadcast
    together (N,)(N+1,)" tramite zero-padding dei due array RIR (ISM e RT)
    al massimo, con fallback finale a ISM puro se la patch non basta.
  - Restituzione di RIR_HF come array numpy 1D normalizzato a picco unitario.

Input:
  - RoomAcousticProfile (Modulo 1)
  - Posizione sorgente e microfono in metri (snap libero, PRA fa interp interna)
  - Sample rate (default 44100 Hz)

Output:
  - np.ndarray 1D float64, normalizzato a |peak| = 1

Autore  : Senior Audio DSP / Acoustic Simulation Engineer
Versione: 1.0.0
Python  : >= 3.10
Dipendenze: numpy, pyroomacoustics
=============================================================================
"""

from __future__ import annotations

import logging
from typing import Dict, Final, Tuple

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# pyroomacoustics — import esplicito con messaggio chiaro se mancante
# ---------------------------------------------------------------------------
try:
    import pyroomacoustics as pra
except ImportError as _pra_err:
    raise ImportError(
        "pyroomacoustics non trovato. Installalo con:\n"
        "    pip install pyroomacoustics\n"
        f"Errore originale: {_pra_err}"
    ) from _pra_err

# ---------------------------------------------------------------------------
# Import Modulo 1
# ---------------------------------------------------------------------------
from m1_physics_setup import RoomAcousticProfile, SURFACE_KEYS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Costanti — replicano la configurazione validata del vecchio script
# ---------------------------------------------------------------------------
DEFAULT_FS:            Final[int]   = 44100
DEFAULT_MAX_ORDER:     Final[int]   = 15        # ordine ISM
DEFAULT_N_RAYS:        Final[int]   = 10000     # raggi per RT
DEFAULT_ENERGY_THRES:  Final[float] = 1e-5
DEFAULT_RX_RADIUS:     Final[float] = 0.1       # raggio microfono [m]
DEFAULT_SCATTERING:    Final[float] = 0.15      # come da istruzioni Modulo 3
ALPHA_CLIP_MIN:        Final[float] = 0.02      # PRA è instabile con alpha < 0.02
ALPHA_CLIP_MAX:        Final[float] = 0.99

# Bande d'ottava attese dai materiali PRA (deve corrispondere a center_freqs)
PRA_CENTER_FREQS:      Final[Tuple[int, ...]] = (125, 250, 500, 1000, 2000, 4000)

# Mapping tra i nomi delle superfici del Modulo 1 e i nomi attesi da pra.ShoeBox.
# pra.ShoeBox accetta un dict materials con le chiavi:
#   floor, ceiling, east, west, north, south.
# Il Modulo 1 usa: floor, ceiling, wall_front, wall_back, wall_left, wall_right.
#
# Convenzione assunta (coerente con il vecchio script):
#   wall_front -> "north"   (x = L  -> alta lunghezza)
#   wall_back  -> "south"   (x = 0)
#   wall_right -> "east"    (y = W)
#   wall_left  -> "west"    (y = 0)
SURFACE_MAPPING: Final[Dict[str, str]] = {
    "floor":      "floor",
    "ceiling":    "ceiling",
    "wall_front": "north",
    "wall_back":  "south",
    "wall_right": "east",
    "wall_left":  "west",
}


# ---------------------------------------------------------------------------
# Eccezione custom
# ---------------------------------------------------------------------------

class PyroomSimulationError(RuntimeError):
    """
    Sollevata quando la simulazione pyroomacoustics fallisce anche dopo
    il fallback a ISM puro.
    """
    pass


# ---------------------------------------------------------------------------
# PyroomEngine
# ---------------------------------------------------------------------------

class PyroomEngine:
    """
    Wrapper OOP del motore pyroomacoustics per la generazione della
    RIR ad alta frequenza (RIR_HF) tramite ISM + Ray Tracing.

    Uso tipico
    ----------
    >>> engine = PyroomEngine()
    >>> rir_hf = engine.run_simulation(
    ...     profile=room_profile,
    ...     src_pos_m=(1.0, 1.0, 1.0),
    ...     mic_pos_m=(4.0, 3.0, 1.5),
    ...     fs=44100,
    ... )

    Notes
    -----
    Tre livelli di robustezza nell'ordine:
      1. Tentativo normale: ISM(max_order=15) + RT(n_rays=10000)
      2. Patch zero-padding: somma manuale di rir_ISM e rir_rt60 con
         padding al massimo se compute_rir() solleva il ValueError di
         broadcast (bug noto di PRA su array di lunghezza ±1).
      3. Fallback ISM puro: ricostruzione della ShoeBox senza Ray Tracing.
         Trade-off: meno fedele in coda (no scattering diffuso) ma stabile.
    """

    def __init__(
        self,
        max_order:        int   = DEFAULT_MAX_ORDER,
        n_rays:           int   = DEFAULT_N_RAYS,
        energy_threshold: float = DEFAULT_ENERGY_THRES,
        receiver_radius:  float = DEFAULT_RX_RADIUS,
        scattering:       float = DEFAULT_SCATTERING,
    ) -> None:
        """
        Parameters
        ----------
        max_order : int
            Ordine massimo dell'Image Source Method. Default = 15.
        n_rays : int
            Numero di raggi per il Ray Tracing. Default = 10000.
        energy_threshold : float
            Soglia di terminazione per i raggi. Default = 1e-5.
        receiver_radius : float
            Raggio del microfono virtuale [m]. Default = 0.1.
        scattering : float
            Coefficiente di scattering diffuso uniforme. Default = 0.15.
        """
        if max_order < 1:
            raise ValueError(f"max_order >= 1 richiesto, ricevuto {max_order}.")
        if n_rays < 100:
            raise ValueError(f"n_rays >= 100 richiesto, ricevuto {n_rays}.")
        if not (0.0 <= scattering <= 1.0):
            raise ValueError(
                f"scattering deve essere in [0, 1], ricevuto {scattering}."
            )

        self.max_order        = max_order
        self.n_rays           = n_rays
        self.energy_threshold = energy_threshold
        self.receiver_radius  = receiver_radius
        self.scattering       = scattering

        logger.info(
            "PyroomEngine inizializzato: max_order=%d, n_rays=%d, "
            "energy_thres=%.1e, rx_radius=%.3f m, scattering=%.2f",
            max_order, n_rays, energy_threshold, receiver_radius, scattering,
        )

    # ------------------------------------------------------------------ #
    #  Conversione profilo → materiali PRA                                 #
    # ------------------------------------------------------------------ #

    def _build_materials(
        self,
        profile: RoomAcousticProfile,
    ) -> Dict[str, "pra.Material"]:
        """
        Converte i coefficienti alpha del RoomAcousticProfile in un
        dizionario di pra.Material indicizzato per superficie PRA
        (floor, ceiling, east, west, north, south).

        Strategia
        ---------
        Per ogni superficie del Modulo 1:
          1. Estrae i coefficienti alpha per le 6 bande standard PRA
             (125, 250, 500, 1000, 2000, 4000). Se una banda manca,
             usa il valore della banda più vicina come fallback.
          2. Clippa i valori a [0.02, 0.99] — PRA è numericamente
             instabile con alpha < 0.02 (problemi di propagazione raggi
             quasi-perfettamente riflessi).
          3. Costruisce pra.Material con energy_absorption per-banda +
             scattering uniforme.
          4. Mappa il nome del Modulo 1 al nome PRA tramite SURFACE_MAPPING.

        Parameters
        ----------
        profile : RoomAcousticProfile

        Returns
        -------
        Dict[str, pra.Material]
            Chiavi: floor, ceiling, east, west, north, south.
        """
        materials: Dict[str, "pra.Material"] = {}

        for m1_surface in SURFACE_KEYS:
            bands = profile.alpha_per_surface[m1_surface]

            # Estrai alpha per le 6 bande PRA (con fallback alla più vicina)
            coeffs_raw = []
            for freq in PRA_CENTER_FREQS:
                if freq in bands:
                    alpha = bands[freq]
                else:
                    closest = min(bands.keys(), key=lambda f: abs(f - freq))
                    alpha = bands[closest]
                    logger.debug(
                        "Surface '%s' banda %d Hz mancante, fallback a %d Hz (alpha=%.3f).",
                        m1_surface, freq, closest, alpha,
                    )
                coeffs_raw.append(alpha)

            # Clipping di sicurezza per PRA
            coeffs_clipped = [
                float(np.clip(a, ALPHA_CLIP_MIN, ALPHA_CLIP_MAX))
                for a in coeffs_raw
            ]

            if coeffs_clipped != coeffs_raw:
                logger.debug(
                    "Surface '%s' alpha clippato in [%.2f, %.2f]: "
                    "%s -> %s",
                    m1_surface, ALPHA_CLIP_MIN, ALPHA_CLIP_MAX,
                    coeffs_raw, coeffs_clipped,
                )

            pra_surface = SURFACE_MAPPING[m1_surface]
            materials[pra_surface] = pra.Material(
                energy_absorption={
                    "description":  f"M1_{m1_surface}",
                    "coeffs":       coeffs_clipped,
                    "center_freqs": list(PRA_CENTER_FREQS),
                },
                scattering=self.scattering,
            )

        logger.info(
            "Materiali PRA costruiti per %d superfici (scattering=%.2f).",
            len(materials), self.scattering,
        )
        return materials

    # ------------------------------------------------------------------ #
    #  Validazione coordinate                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_position(
        pos: Tuple[float, float, float],
        room_dims: Tuple[float, float, float],
        label: str,
    ) -> None:
        """
        Verifica che (x, y, z) cadano strettamente dentro la stanza,
        con margine di 0.05 m dalle pareti (PRA può comportarsi male se
        sorgente/microfono sono troppo vicini ai muri).
        """
        L, W, H = room_dims
        x, y, z = pos
        margin: float = 0.05

        if not (margin <= x <= L - margin and
                margin <= y <= W - margin and
                margin <= z <= H - margin):
            raise ValueError(
                f"{label} pos {pos} m fuori dalla stanza "
                f"({L:.2f} x {W:.2f} x {H:.2f} m) con margine {margin} m. "
                f"Riposiziona almeno a {margin} m dalle pareti."
            )

    # ------------------------------------------------------------------ #
    #  Costruzione ShoeBox                                                 #
    # ------------------------------------------------------------------ #

    def _build_shoebox(
        self,
        dims:      Tuple[float, float, float],
        materials: Dict[str, "pra.Material"],
        fs:        int,
        ray_tracing: bool,
    ) -> "pra.ShoeBox":
        """
        Crea una pra.ShoeBox con la configurazione ibrida ISM + Ray Tracing
        (oppure ISM puro se ray_tracing=False, usato come fallback).
        """
        room = pra.ShoeBox(
            list(dims),
            materials=materials,
            fs=fs,
            max_order=self.max_order,
            ray_tracing=ray_tracing,
            air_absorption=True,
        )

        if ray_tracing:
            room.set_ray_tracing(
                receiver_radius=self.receiver_radius,
                n_rays=self.n_rays,
                energy_thres=self.energy_threshold,
            )

        return room

    # ------------------------------------------------------------------ #
    #  Patch zero-padding per il bug di broadcast PRA                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _patch_broadcast_bug(room: "pra.ShoeBox") -> bool:
        """
        Patch per il noto bug PRA:
            "operands could not be broadcast together (N,)(N+1,)"

        Causa: dentro compute_rir() PRA somma rir_ISM e rir_rt60 ma i due
        array hanno lunghezze disallineate di esattamente 1 sample, e
        numpy rifiuta il broadcast.

        Fix: zero-padding manuale di entrambi gli array alla lunghezza
        massima e somma esplicita, sostituendo room.rir[mic][src] con
        il risultato. La fisica della simulazione non viene toccata:
        si aggiusta solo il buffer.

        Returns
        -------
        bool
            True se la patch è riuscita, False se i campi rir_ISM/rir_rt60
            non sono accessibili (caso in cui serve il fallback ISM puro).
        """
        rir_ism = getattr(room, "rir_ISM",  None)
        rir_rt  = getattr(room, "rir_rt60", None)

        if rir_ism is None or rir_rt is None:
            logger.warning(
                "patch_broadcast_bug: room.rir_ISM o room.rir_rt60 non accessibili. "
                "La versione di PRA in uso non espone questi attributi. "
                "Procedo con fallback ISM puro."
            )
            return False

        n_mics    = len(room.rir)
        n_sources = len(room.rir[0]) if n_mics else 0

        for m in range(n_mics):
            for s in range(n_sources):
                a = np.asarray(rir_ism[m][s], dtype=np.float64)
                b = np.asarray(rir_rt[m][s],  dtype=np.float64)
                n = max(len(a), len(b))
                a_pad = np.pad(a, (0, n - len(a)))
                b_pad = np.pad(b, (0, n - len(b)))
                room.rir[m][s] = a_pad + b_pad

        logger.info(
            "patch_broadcast_bug: zero-padding applicato a %d mic x %d sorgenti.",
            n_mics, n_sources,
        )
        return True

    # ------------------------------------------------------------------ #
    #  Entry point pubblico                                                #
    # ------------------------------------------------------------------ #

    def run_simulation(
        self,
        profile:   RoomAcousticProfile,
        src_pos_m: Tuple[float, float, float],
        mic_pos_m: Tuple[float, float, float],
        fs:        int = DEFAULT_FS,
    ) -> NDArray[np.float64]:
        """
        Esegue la simulazione geometrica ISM + Ray Tracing e restituisce
        la RIR_HF normalizzata.

        Sequenza
        --------
        1. Valida le posizioni (margine dalle pareti).
        2. Costruisce i materiali PRA con clipping alpha.
        3. Tenta la simulazione ibrida ISM+RT.
        4. Se compute_rir() solleva "operands could not be broadcast",
           applica la patch zero-padding.
        5. Se la patch fallisce (attributi PRA non esposti), ricostruisce
           la ShoeBox senza Ray Tracing ed esegue ISM puro.
        6. Normalizza la RIR al picco unitario e la restituisce.

        Parameters
        ----------
        profile : RoomAcousticProfile
        src_pos_m : Tuple[float, float, float]
            (x, y, z) sorgente [m].
        mic_pos_m : Tuple[float, float, float]
            (x, y, z) microfono [m].
        fs : int
            Sample rate [Hz]. Default = 44100.

        Returns
        -------
        NDArray[np.float64]
            RIR_HF, shape (N,), normalizzata a |peak| = 1.

        Raises
        ------
        PyroomSimulationError
            Se anche il fallback ISM puro fallisce.
        """
        dims = (profile.length, profile.width, profile.height)

        self._validate_position(src_pos_m, dims, "sorgente")
        self._validate_position(mic_pos_m, dims, "microfono")

        logger.info("=" * 60)
        logger.info("PyroomEngine.run_simulation — START")
        logger.info(
            "  Stanza: %.2f x %.2f x %.2f m | fs=%d Hz",
            dims[0], dims[1], dims[2], fs,
        )
        logger.info("  Source: %s m | Mic: %s m", src_pos_m, mic_pos_m)
        logger.info("=" * 60)

        materials = self._build_materials(profile)

        # ---------------- Stadio 1: ISM + Ray Tracing ----------------- #
        room = self._build_shoebox(dims, materials, fs, ray_tracing=True)
        room.add_source(list(src_pos_m))
        room.add_microphone(list(mic_pos_m))

        try:
            logger.info("Stadio 1: compute_rir() con ISM+RT...")
            room.compute_rir()
            logger.info("Stadio 1: OK.")

        except ValueError as rir_err:
            if "operands could not be broadcast" not in str(rir_err):
                # ValueError diverso da quello noto - non lo catturiamo
                raise

            logger.warning(
                "Bug broadcast PRA rilevato: '%s'. Tentativo patch zero-padding...",
                rir_err,
            )

            # ----------- Stadio 2: patch zero-padding ----------------- #
            patched = self._patch_broadcast_bug(room)

            if not patched:
                # ----------- Stadio 3: fallback ISM puro -------------- #
                logger.warning(
                    "Patch fallita. Fallback a ISM puro (ray_tracing=False)."
                )
                try:
                    room = self._build_shoebox(
                        dims, materials, fs, ray_tracing=False
                    )
                    room.add_source(list(src_pos_m))
                    room.add_microphone(list(mic_pos_m))
                    room.compute_rir()
                    logger.info("Stadio 3: fallback ISM puro completato.")
                except Exception as e:
                    raise PyroomSimulationError(
                        f"Fallback ISM puro fallito: {type(e).__name__}: {e}"
                    ) from e

        except Exception as e:
            raise PyroomSimulationError(
                f"compute_rir() ha sollevato {type(e).__name__}: {e}"
            ) from e

        # ---------------- Estrazione RIR ------------------------------ #
        rir_raw = np.asarray(room.rir[0][0], dtype=np.float64)

        if rir_raw.size == 0:
            raise PyroomSimulationError(
                "room.rir[0][0] è vuoto. La simulazione non ha prodotto output."
            )

        peak = float(np.max(np.abs(rir_raw)))
        if peak < 1e-15:
            logger.warning(
                "RIR_HF picco quasi nullo (%.2e). Probabile problema fisico.",
                peak,
            )
            rir_norm = rir_raw  # niente normalizzazione su segnale nullo
        else:
            rir_norm = rir_raw / peak

        logger.info(
            "PyroomEngine.run_simulation DONE — RIR_HF: %d samples | "
            "peak originale=%.4e | durata=%.3f s",
            len(rir_norm), peak, len(rir_norm) / fs,
        )
        return rir_norm


# ---------------------------------------------------------------------------
# Funzione di convenienza
# ---------------------------------------------------------------------------

def generate_rir_hf(
    profile:   RoomAcousticProfile,
    src_pos_m: Tuple[float, float, float],
    mic_pos_m: Tuple[float, float, float],
    fs:        int = DEFAULT_FS,
) -> NDArray[np.float64]:
    """
    Funzione di convenienza per l'uso diretto dalla pipeline (Modulo 4).
    Istanzia un PyroomEngine con i default validati ed esegue una singola
    simulazione.

    Parameters
    ----------
    profile : RoomAcousticProfile
    src_pos_m, mic_pos_m : Tuple[float, float, float]
    fs : int

    Returns
    -------
    NDArray[np.float64]
        RIR_HF normalizzata.
    """
    engine = PyroomEngine()
    return engine.run_simulation(profile, src_pos_m, mic_pos_m, fs)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n" + "=" * 60)
    print("MODULO 3 — Self Test")
    print("=" * 60)

    # Stanza di test (stessa del Modulo 1)
    room = RoomAcousticProfile(
        length=6.0,
        width=4.0,
        height=3.0,
        alpha_per_surface={
            "floor":      {125: 0.02, 250: 0.03, 500: 0.05, 1000: 0.07, 2000: 0.08, 4000: 0.08},
            "ceiling":    {125: 0.15, 250: 0.20, 500: 0.25, 1000: 0.30, 2000: 0.35, 4000: 0.40},
            "wall_front": {125: 0.05, 250: 0.07, 500: 0.10, 1000: 0.12, 2000: 0.13, 4000: 0.15},
            "wall_back":  {125: 0.40, 250: 0.45, 500: 0.50, 1000: 0.50, 2000: 0.55, 4000: 0.60},
            "wall_left":  {125: 0.05, 250: 0.07, 500: 0.10, 1000: 0.12, 2000: 0.13, 4000: 0.15},
            "wall_right": {125: 0.05, 250: 0.07, 500: 0.10, 1000: 0.12, 2000: 0.13, 4000: 0.15},
        },
    )

    print(f"\nStanza: {room}")

    engine = PyroomEngine()
    rir_hf = engine.run_simulation(
        profile=room,
        src_pos_m=(1.0, 1.0, 1.0),
        mic_pos_m=(4.5, 3.0, 1.5),
        fs=DEFAULT_FS,
    )

    print(f"\nRIR_HF prodotta:")
    print(f"  shape       = {rir_hf.shape}")
    print(f"  duration    = {len(rir_hf) / DEFAULT_FS:.3f} s")
    print(f"  peak        = {float(np.max(np.abs(rir_hf))):.6f}")
    print(f"  argmax      = sample {int(np.argmax(np.abs(rir_hf)))} "
          f"({int(np.argmax(np.abs(rir_hf))) / DEFAULT_FS * 1000:.2f} ms)")

    print("\n" + "=" * 60)
    print("Self test Modulo 3 completato.")
    print("=" * 60)