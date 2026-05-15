"""
=============================================================================
MODULO 1 — Physics Setup & Parameter Translation
=============================================================================
Simulatore Acustico Ibrido XR — HPC-Ready Pipeline

Responsabilità:
  - Definizione del profilo acustico della stanza (RoomAcousticProfile).
  - Calcolo della Frequenza di Schroeder (f_s) come punto di crossover.
  - Dimensionamento della griglia di voxel k-Wave (dx, Nx, Ny, Nz).
  - Traduzione dei coefficienti di assorbimento (alpha) in densità (rho)
    per i boundary FDTD di k-Wave.
  - Guardrail preventivo sulla RAM stimata prima di lanciare k-Wave.

Autore  : Senior Audio DSP / Acoustic Simulation Engineer
Versione: 1.0.0
Python  : >= 3.10
Dipendenze: numpy, dataclasses (stdlib), typing (stdlib), logging (stdlib)
=============================================================================
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Final, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Costanti fisiche globali
# ---------------------------------------------------------------------------
C_AIR: Final[float] = 343.0       # Velocità del suono in aria [m/s] @ 20 °C
Z0_AIR: Final[float] = 415.0      # Impedenza caratteristica aria [Rayl = Pa·s/m]
RHO_AIR: Final[float] = 1.21      # Densità aria standard [kg/m³]
PPW: Final[int] = 4                # Points-Per-Wavelength per accuratezza FDTD
BYTES_PER_VOXEL: Final[int] = 40   # ~40 byte/voxel (10 array float32 k-Wave)
F_SCHROEDER_CAP: Final[float] = 1000.0  # Limite superiore di sicurezza f_s [Hz]

# Superfici standard attese nel dizionario alpha
SURFACE_KEYS: Final[Tuple[str, ...]] = (
    "floor", "ceiling", "wall_front", "wall_back", "wall_left", "wall_right"
)


# ---------------------------------------------------------------------------
# Eccezione custom
# ---------------------------------------------------------------------------

class MemoryConstraintError(MemoryError):
    """
    Sollevata da PhysicsTranslator.ram_guardrail quando la griglia k-Wave
    stimata supera il budget di RAM consentito.

    Attributes
    ----------
    required_gb : float
        RAM stimata necessaria [GB].
    max_gb : float
        Limite massimo configurato [GB].
    grid_shape : Tuple[int, int, int]
        Dimensioni della griglia (Nx, Ny, Nz) che causano il problema.
    """

    def __init__(
        self,
        required_gb: float,
        max_gb: float,
        grid_shape: Tuple[int, int, int],
    ) -> None:
        self.required_gb = required_gb
        self.max_gb = max_gb
        self.grid_shape = grid_shape
        super().__init__(
            f"Griglia k-Wave {grid_shape[0]}x{grid_shape[1]}x{grid_shape[2]} "
            f"richiederebbe ~{required_gb:.2f} GB > limite {max_gb:.2f} GB. "
            f"Aumenta dx, riduci la stanza o incrementa max_ram_gb."
        )


# ---------------------------------------------------------------------------
# Dataclass — profilo acustico della stanza
# ---------------------------------------------------------------------------

@dataclass
class RoomAcousticProfile:
    """
    Profilo acustico completo di una stanza rettangolare.

    Contiene sia la geometria fisica che i coefficienti di assorbimento
    per banda d'ottava di ciascuna delle 6 superfici, più i parametri
    derivati (volume, T60 stimato) utilizzati a valle dai moduli k-Wave
    e pyroomacoustics.

    Parameters
    ----------
    length : float
        Dimensione X della stanza [m].
    width : float
        Dimensione Y della stanza [m].
    height : float
        Dimensione Z della stanza [m].
    alpha_per_surface : dict
        Dizionario annidato:
          { superficie: { freq_hz: alpha } }
        es. {"floor": {125: 0.02, 250: 0.03}, "ceiling": {125: 0.05}, ...}
        Le superfici attese sono: floor, ceiling, wall_front, wall_back,
        wall_left, wall_right.
    t60_target : float | None
        T60 misurato/target [s]. Se None viene stimato con Sabine.

    Attributes derivati (calcolati in __post_init__)
    ------------------------------------------------
    volume : float
        Volume della stanza [m³].
    total_surface : float
        Superficie totale delle 6 pareti [m²].
    t60_sabine : float
        T60 stimato con la formula di Sabine [s].
    """

    length: float
    width:  float
    height: float
    alpha_per_surface: Dict[str, Dict[int, float]]
    t60_target: float | None = None

    # Campi derivati — calcolati in __post_init__
    volume:        float = field(init=False)
    total_surface: float = field(init=False)
    t60_sabine:    float = field(init=False)

    # ------------------------------------------------------------------ #
    #  Validazione e calcoli derivati                                      #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        self._validate_geometry()
        self._validate_alpha_dict()
        self.volume        = self._compute_volume()
        self.total_surface = self._compute_total_surface()
        self.t60_sabine    = self._compute_t60_sabine()

    def _validate_geometry(self) -> None:
        for attr, val in [
            ("length", self.length),
            ("width",  self.width),
            ("height", self.height),
        ]:
            if not isinstance(val, (int, float)):
                raise TypeError(
                    f"RoomAcousticProfile.{attr} deve essere float, "
                    f"ricevuto {type(val).__name__}."
                )
            if val <= 0.0:
                raise ValueError(
                    f"RoomAcousticProfile.{attr} = {val} deve essere > 0."
                )
            if val > 30.0:
                warnings.warn(
                    f"RoomAcousticProfile.{attr} = {val} m sembra insolitamente "
                    f"grande (> 30 m). Verifica le unità di misura.",
                    UserWarning,
                    stacklevel=3,
                )

    def _validate_alpha_dict(self) -> None:
        for surface in SURFACE_KEYS:
            if surface not in self.alpha_per_surface:
                raise KeyError(
                    f"alpha_per_surface manca della superficie '{surface}'. "
                    f"Superfici attese: {SURFACE_KEYS}."
                )
        for surface, bands in self.alpha_per_surface.items():
            for freq, alpha in bands.items():
                if not (0.0 <= alpha <= 1.0):
                    raise ValueError(
                        f"alpha_per_surface['{surface}'][{freq}] = {alpha} "
                        f"fuori range [0.0, 1.0]."
                    )

    def _compute_volume(self) -> float:
        return self.length * self.width * self.height

    def _compute_total_surface(self) -> float:
        lx, ly, lz = self.length, self.width, self.height
        return 2.0 * (lx * ly + lx * lz + ly * lz)

    def _compute_t60_sabine(self) -> float:
        """
        Stima T60 con la formula di Sabine:

            T60 = 0.161 * V / sum_i(alpha_i * S_i)

        alpha_i è la media aritmetica sulle bande disponibili per ciascuna
        superficie (proxy broadband sufficiente per la stima Sabine).

        Returns
        -------
        float
            T60 stimato [s], clampato nell'intervallo [0.05, 20.0].
        """
        lx, ly, lz = self.length, self.width, self.height
        surface_areas: Dict[str, float] = {
            "floor":      lx * ly,
            "ceiling":    lx * ly,
            "wall_front": lx * lz,
            "wall_back":  lx * lz,
            "wall_left":  ly * lz,
            "wall_right": ly * lz,
        }

        total_absorption: float = 0.0
        for surface, area in surface_areas.items():
            bands = self.alpha_per_surface.get(surface, {})
            if not bands:
                logger.warning(
                    "Superficie '%s' senza bande alpha — uso alpha=0.1 come fallback.",
                    surface,
                )
                alpha_mean = 0.1
            else:
                alpha_mean = float(np.mean(list(bands.values())))
            total_absorption += alpha_mean * area

        if total_absorption <= 0.0:
            raise ValueError(
                "Assorbimento totale Sabine = 0. "
                "Almeno una superficie deve avere alpha > 0."
            )

        t60_raw: float = 0.161 * self.volume / total_absorption
        t60_clamped: float = float(np.clip(t60_raw, 0.05, 20.0))

        if abs(t60_clamped - t60_raw) > 1e-6:
            logger.warning(
                "T60 Sabine raw=%.4f s clampato a %.4f s (range [0.05, 20.0]).",
                t60_raw, t60_clamped,
            )

        logger.info(
            "T60 Sabine: 0.161 * %.2f / %.4f = %.4f s",
            self.volume, total_absorption, t60_clamped,
        )
        return t60_clamped

    # ------------------------------------------------------------------ #
    #  Proprietà pubblica                                                  #
    # ------------------------------------------------------------------ #

    @property
    def t60(self) -> float:
        """
        T60 effettivo: usa t60_target se fornito, altrimenti t60_sabine.
        Questo è il valore propagato a tutti i calcoli downstream.
        """
        return self.t60_target if self.t60_target is not None else self.t60_sabine

    def __repr__(self) -> str:
        return (
            f"RoomAcousticProfile("
            f"L={self.length:.2f}m x W={self.width:.2f}m x H={self.height:.2f}m | "
            f"V={self.volume:.2f}m3 | T60={self.t60:.3f}s)"
        )


# ---------------------------------------------------------------------------
# Classe statica — traduttore fisico
# ---------------------------------------------------------------------------

class PhysicsTranslator:
    """
    Gestore statico per la traduzione dei parametri fisici della stanza
    nei parametri numerici richiesti dai motori k-Wave e pyroomacoustics.

    Tutti i metodi sono @staticmethod: non mantengono stato interno e
    possono essere chiamati senza istanziare la classe.

    Metodi pubblici
    ---------------
    get_schroeder_frequency  : Calcola la frequenza di Schroeder f_s [Hz].
    get_voxel_size           : Calcola il passo spaziale dx per la griglia FDTD.
    alpha_to_density         : Traduce alpha -> rho_wall per k-Wave boundary.
    ram_guardrail            : Stima la RAM e solleva MemoryConstraintError se
                               si supera il budget consentito.
    translate_profile        : Pipeline completa: RoomAcousticProfile -> dict
                               con tutti i parametri pronti per il Modulo 2.
    """

    # ------------------------------------------------------------------ #
    #  1. Frequenza di Schroeder                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_schroeder_frequency(profile: RoomAcousticProfile) -> float:
        """
        Calcola la Frequenza di Schroeder f_s, che separa la zona modale
        (basse frequenze, wave-based FDTD) dalla zona geometrica
        (alte frequenze, ray-tracing).

        Formula
        -------
            f_s = 2000 * sqrt(T60 / V)

        dove:
          - T60 è il tempo di riverberazione effettivo [s]  (profile.t60)
          - V   è il volume della stanza [m3]

        Limite di sicurezza
        -------------------
        f_s viene cappata a F_SCHROEDER_CAP (1000 Hz) per evitare che
        stanze molto piccole o molto riflettenti generino frequenze di
        crossover eccessive, causando esplosione combinatoria della griglia
        FDTD (dx proporzionale a 1/f_s).

        Parameters
        ----------
        profile : RoomAcousticProfile

        Returns
        -------
        float
            Frequenza di Schroeder f_s [Hz], cappata a F_SCHROEDER_CAP.

        Raises
        ------
        ValueError
            Se T60 o volume risultano non positivi.
        """
        t60    = profile.t60
        volume = profile.volume

        if t60 <= 0.0 or volume <= 0.0:
            raise ValueError(
                f"T60={t60:.4f} s e Volume={volume:.4f} m3 devono essere > 0 "
                f"per il calcolo di Schroeder."
            )

        f_s_raw: float = 2000.0 * math.sqrt(t60 / volume)
        logger.info(
            "Schroeder: f_s = 2000 x sqrt(%.4f / %.4f) = %.2f Hz",
            t60, volume, f_s_raw,
        )

        if f_s_raw > F_SCHROEDER_CAP:
            logger.warning(
                "f_s calcolata (%.2f Hz) supera il limite di sicurezza "
                "(%.0f Hz). Cappata per evitare esplosione FDTD. "
                "Valuta stanza piu' grande o materiali piu' assorbenti.",
                f_s_raw, F_SCHROEDER_CAP,
            )
            return F_SCHROEDER_CAP

        return f_s_raw

    # ------------------------------------------------------------------ #
    #  2. Dimensione del voxel k-Wave                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_voxel_size(
        f_s: float,
        ppw: int = PPW,
        c: float = C_AIR,
    ) -> float:
        """
        Calcola il passo spaziale dx per la griglia FDTD di k-Wave.

        Formula
        -------
            lambda_min = c / f_s
            dx = lambda_min / PPW

        Con PPW = 4 si garantisce che la lunghezza d'onda minima simulata
        contenga almeno 4 punti di griglia, soddisfacendo il criterio di
        Nyquist spaziale per la stabilità dell'FDTD.

        Parameters
        ----------
        f_s : float
            Frequenza massima simulata da k-Wave (= f_s Schroeder) [Hz].
        ppw : int
            Points-Per-Wavelength. Default = 4.
        c : float
            Velocità del suono [m/s]. Default = 343.0.

        Returns
        -------
        float
            Dimensione massima del voxel dx [m].

        Raises
        ------
        ValueError
            Se f_s <= 0 o ppw < 2.
        """
        if f_s <= 0.0:
            raise ValueError(f"f_s deve essere > 0, ricevuto {f_s}.")
        if ppw < 2:
            raise ValueError(
                f"PPW deve essere >= 2 per stabilità FDTD, ricevuto {ppw}."
            )

        lambda_min: float = c / f_s
        dx: float         = lambda_min / ppw

        logger.info(
            "Voxel size: lambda_min = c/f_s = %.4f m -> dx = %.4f m (%.2f cm) [PPW=%d]",
            lambda_min, dx, dx * 100, ppw,
        )
        return dx

    # ------------------------------------------------------------------ #
    #  3. Traduzione alpha -> densità muro                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def alpha_to_density(
        alpha: float,
        z0: float = Z0_AIR,
        c:  float = C_AIR,
    ) -> Tuple[float, float, float]:
        """
        Traduce il coefficiente di assorbimento alpha di una superficie in
        densità acustica rho_wall, per i boundary FDTD di k-Wave.

        Pipeline di calcolo
        -------------------

        Step 1 — Coefficiente di riflessione in ampiezza:

            |R| = sqrt(1 - alpha)

            alpha = 1 - |R|^2  (definizione energia), quindi |R| = sqrt(1 - alpha).
            alpha=0 (muro rigido)    -> |R|=1  (riflessione totale)
            alpha=1 (assorbente tot) -> |R|=0  (nessuna riflessione)

        Step 2 — Impedenza acustica del muro (incidenza normale):

            Z_wall = Z0 * (1 + |R|) / (1 - |R|)

            Segue dalla definizione di coefficiente di riflessione di pressione
            all'interfaccia tra due mezzi:
                R = (Z2 - Z1) / (Z2 + Z1)  =>  Z2 = Z1 * (1+R)/(1-R)

            Caso limite |R| -> 1 (muro rigido):    Z_wall -> infinito.
            Caso limite |R| = 0 (anecoico):        Z_wall = Z0 (matching perfetto).

        Step 3 — Densità fittizia del voxel boundary (Z = rho * c):

            rho_wall = Z_wall / c

            Si assume c costante nel voxel boundary = c_aria.
            L'intera variazione di impedenza e' concentrata su rho_wall.
            Questo e' il parametro assegnato ai voxel di bordo in k-Wave.

        Parameters
        ----------
        alpha : float
            Coefficiente di assorbimento in [0.0, 1.0].
        z0 : float
            Impedenza caratteristica aria [Rayl]. Default = 415.
        c : float
            Velocità del suono [m/s]. Default = 343.

        Returns
        -------
        Tuple[float, float, float]
            (reflection_coeff, Z_wall, rho_wall)
            - reflection_coeff : |R| in [0, 1]
            - Z_wall           : impedenza muro [Rayl]
            - rho_wall         : densità fittizia boundary [kg/m3]

        Raises
        ------
        ValueError
            Se alpha e' fuori [0.0, 1.0].
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(
                f"alpha={alpha} fuori range [0.0, 1.0]. "
                f"Verifica i dati del materiale."
            )

        # Clipping |R| per evitare singolarità nel denominatore (1 - |R|)
        # quando alpha -> 0 (muro quasi-rigido). Il valore 0.9999 corrisponde
        # ad alpha_min ≈ 2e-4, fisicamente raggiungibile solo da acciaio/cemento.
        MAX_R: Final[float] = 0.9999

        reflection_coeff: float = math.sqrt(max(0.0, 1.0 - alpha))
        reflection_coeff = min(reflection_coeff, MAX_R)

        if reflection_coeff >= MAX_R:
            logger.debug(
                "alpha=%.5f: |R| clampato a %.4f (muro quasi-rigido, Z_wall elevata).",
                alpha, MAX_R,
            )

        Z_wall:   float = z0 * (1.0 + reflection_coeff) / (1.0 - reflection_coeff)
        rho_wall: float = Z_wall / c

        logger.debug(
            "alpha=%.4f -> |R|=%.4f -> Z_wall=%.1f Rayl -> rho_wall=%.4f kg/m3",
            alpha, reflection_coeff, Z_wall, rho_wall,
        )
        return reflection_coeff, Z_wall, rho_wall

    # ------------------------------------------------------------------ #
    #  4. Guardrail RAM                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def ram_guardrail(
        dx:        float,
        length:    float,
        width:     float,
        height:    float,
        max_ram_gb: float,
    ) -> Tuple[int, int, int, float]:
        """
        Stima la RAM richiesta dalla griglia k-Wave e lancia
        MemoryConstraintError se supera il budget consentito.

        Modello di stima RAM
        --------------------
        k-Wave alloca tipicamente 8-10 array 3D float32:
          pressione p, velocità ux/uy/uz, densità rho, velocità c,
          + buffer PML e array temporanei.

            RAM_bytes = Nx * Ny * Nz * BYTES_PER_VOXEL

        con BYTES_PER_VOXEL = 40 (conservativo: 10 array * 4 byte float32).

        Dimensioni griglia
        ------------------
            Nx = ceil(length / dx)
            Ny = ceil(width  / dx)
            Nz = ceil(height / dx)

        Parameters
        ----------
        dx : float
            Passo spaziale del voxel [m].
        length, width, height : float
            Dimensioni della stanza [m].
        max_ram_gb : float
            Budget massimo di RAM [GB].

        Returns
        -------
        Tuple[int, int, int, float]
            (Nx, Ny, Nz, ram_gb_estimated)

        Raises
        ------
        MemoryConstraintError
            Se la RAM stimata supera max_ram_gb.
        ValueError
            Se dx <= 0 o max_ram_gb <= 0.
        """
        if dx <= 0.0:
            raise ValueError(f"dx deve essere > 0, ricevuto {dx:.8f}.")
        if max_ram_gb <= 0.0:
            raise ValueError(f"max_ram_gb deve essere > 0, ricevuto {max_ram_gb}.")

        Nx: int = math.ceil(length / dx)
        Ny: int = math.ceil(width  / dx)
        Nz: int = math.ceil(height / dx)

        total_voxels: int  = Nx * Ny * Nz
        ram_bytes:    float = float(total_voxels) * BYTES_PER_VOXEL
        ram_gb:       float = ram_bytes / (1024.0 ** 3)

        logger.info(
            "RAM guardrail: griglia %d x %d x %d = %s voxels "
            "-> RAM stimata ~%.3f GB (limite: %.2f GB)",
            Nx, Ny, Nz, f"{total_voxels:,}", ram_gb, max_ram_gb,
        )

        if ram_gb > max_ram_gb:
            raise MemoryConstraintError(
                required_gb=ram_gb,
                max_gb=max_ram_gb,
                grid_shape=(Nx, Ny, Nz),
            )

        logger.info(
            "RAM check OK: %.3f GB < %.2f GB — griglia (%d x %d x %d) accettata.",
            ram_gb, max_ram_gb, Nx, Ny, Nz,
        )
        return Nx, Ny, Nz, ram_gb

    # ------------------------------------------------------------------ #
    #  5. Pipeline completa di traduzione                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def translate_profile(
        profile:        RoomAcousticProfile,
        target_freq_hz: int   = 125,
        max_ram_gb:     float = 16.0,
        ppw:            int   = PPW,
    ) -> dict:
        """
        Pipeline completa di traduzione fisica.

        RoomAcousticProfile  ->  dict con tutti i parametri numerici
        pronti per il Modulo 2 (k-Wave FDTD) e il Modulo 4 (Crossover DSP).

        Sequenza di esecuzione
        ----------------------
        1. get_schroeder_frequency  -> f_s
        2. get_voxel_size           -> dx
        3. ram_guardrail            -> (Nx, Ny, Nz, ram_gb)
        4. alpha_to_density per ogni superficie a target_freq_hz
                                   -> rho_per_surface, z_wall_per_surface

        Parameters
        ----------
        profile : RoomAcousticProfile
        target_freq_hz : int
            Banda d'ottava per la traduzione alpha -> rho (tipicamente 125
            o 250 Hz, le bande rilevanti per k-Wave). Default = 125.
        max_ram_gb : float
            Budget RAM per la simulazione k-Wave [GB]. Default = 16.0.
        ppw : int
            Points-Per-Wavelength per il voxel sizing. Default = 4.

        Returns
        -------
        dict
            Chiavi:
            - "f_schroeder"        : float           — frequenza di crossover [Hz]
            - "dx"                 : float           — passo spaziale voxel [m]
            - "grid_shape"         : Tuple[int,int,int] — (Nx, Ny, Nz)
            - "ram_gb_estimated"   : float           — RAM stimata [GB]
            - "rho_per_surface"    : Dict[str,float] — densità boundary [kg/m3]
            - "z_wall_per_surface" : Dict[str,float] — impedenza boundary [Rayl]
            - "t60"                : float           — T60 usato [s]
            - "volume"             : float           — volume stanza [m3]

        Raises
        ------
        MemoryConstraintError
            Se la griglia k-Wave supera max_ram_gb.
        """
        logger.info("=" * 60)
        logger.info("PhysicsTranslator.translate_profile — START")
        logger.info("Profilo: %s", profile)
        logger.info("=" * 60)

        # Step 1 — Frequenza di Schroeder
        f_s: float = PhysicsTranslator.get_schroeder_frequency(profile)

        # Step 2 — Dimensione voxel
        dx: float = PhysicsTranslator.get_voxel_size(f_s, ppw=ppw)

        # Step 3 — Guardrail RAM (lancia MemoryConstraintError se necessario)
        Nx, Ny, Nz, ram_gb = PhysicsTranslator.ram_guardrail(
            dx=dx,
            length=profile.length,
            width=profile.width,
            height=profile.height,
            max_ram_gb=max_ram_gb,
        )

        # Step 4 — Traduzione alpha -> rho per ogni superficie
        rho_per_surface:    Dict[str, float] = {}
        z_wall_per_surface: Dict[str, float] = {}

        for surface in SURFACE_KEYS:
            bands = profile.alpha_per_surface.get(surface, {})

            if target_freq_hz in bands:
                alpha = bands[target_freq_hz]
            elif bands:
                # Fallback: banda d'ottava più vicina disponibile
                closest = min(bands.keys(), key=lambda f: abs(f - target_freq_hz))
                alpha = bands[closest]
                logger.warning(
                    "Superficie '%s': banda %d Hz non trovata, "
                    "uso %d Hz (alpha=%.4f) come fallback.",
                    surface, target_freq_hz, closest, alpha,
                )
            else:
                alpha = 0.1
                logger.warning(
                    "Superficie '%s': nessun dato alpha disponibile. "
                    "Uso alpha=0.1 come fallback.",
                    surface,
                )

            _, z_wall, rho_wall = PhysicsTranslator.alpha_to_density(alpha)
            rho_per_surface[surface]    = rho_wall
            z_wall_per_surface[surface] = z_wall

        result: dict = {
            "f_schroeder":         f_s,
            "dx":                  dx,
            "grid_shape":          (Nx, Ny, Nz),
            "ram_gb_estimated":    ram_gb,
            "rho_per_surface":     rho_per_surface,
            "z_wall_per_surface":  z_wall_per_surface,
            "t60":                 profile.t60,
            "volume":              profile.volume,
        }

        logger.info(
            "translate_profile DONE: f_s=%.2f Hz | dx=%.4f m | "
            "grid=(%d x %d x %d) | RAM~%.3f GB",
            f_s, dx, Nx, Ny, Nz, ram_gb,
        )
        return result


# ---------------------------------------------------------------------------
# Quick self-test — eseguibile come script standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n" + "=" * 60)
    print("MODULO 1 — Self Test")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Stanza di test: sala conferenze (6m x 4m x 3m)                      #
    # ------------------------------------------------------------------ #
    sample_profile = RoomAcousticProfile(
        length=6.0,
        width=4.0,
        height=3.0,
        alpha_per_surface={
            "floor":      {125: 0.02, 250: 0.03, 500: 0.05, 1000: 0.07},
            "ceiling":    {125: 0.15, 250: 0.20, 500: 0.25, 1000: 0.30},
            "wall_front": {125: 0.05, 250: 0.07, 500: 0.10, 1000: 0.12},
            "wall_back":  {125: 0.40, 250: 0.45, 500: 0.50, 1000: 0.50},
            "wall_left":  {125: 0.05, 250: 0.07, 500: 0.10, 1000: 0.12},
            "wall_right": {125: 0.05, 250: 0.07, 500: 0.10, 1000: 0.12},
        },
        t60_target=None,  # Usa stima Sabine
    )

    print(f"\nProfilo stanza: {sample_profile}")
    print(f"  Volume:        {sample_profile.volume:.2f} m3")
    print(f"  Superficie:    {sample_profile.total_surface:.2f} m2")
    print(f"  T60 Sabine:    {sample_profile.t60_sabine:.3f} s")
    print(f"  T60 effettivo: {sample_profile.t60:.3f} s")

    # ------------------------------------------------------------------ #
    # Test translate_profile — caso OK                                     #
    # ------------------------------------------------------------------ #
    print("\n--- Test translate_profile (atteso: OK) ---")
    try:
        params = PhysicsTranslator.translate_profile(
            profile=sample_profile,
            target_freq_hz=125,
            max_ram_gb=16.0,
        )
        print(f"\n  f_schroeder:      {params['f_schroeder']:.2f} Hz")
        print(f"  dx:               {params['dx']*100:.3f} cm")
        print(f"  grid_shape:       {params['grid_shape']}")
        print(f"  ram_gb_estimated: {params['ram_gb_estimated']:.4f} GB")
        print(f"  t60:              {params['t60']:.3f} s")
        print(f"  volume:           {params['volume']:.2f} m3")
        print(f"\n  Densità superfici [kg/m3]:")
        for s, rho in params["rho_per_surface"].items():
            z = params["z_wall_per_surface"][s]
            print(f"    {s:<12}: rho={rho:.4f}, Z={z:.1f} Rayl")

    except MemoryConstraintError as e:
        print(f"\n[ERRORE RAM] {e}")

    # ------------------------------------------------------------------ #
    # Test MemoryConstraintError                                           #
    # Stanza media + alpha molto basso + t60_target alto                   #
    # -> f_s cappata a 1000 Hz -> dx piccolo (~8.6cm) -> griglia densa    #
    # -> RAM ~0.019 GB > budget artificiale di 0.001 GB                   #
    # ------------------------------------------------------------------ #
    print("\n--- Test MemoryConstraintError (atteso: ERRORE) ---")
    # Stanza media con f_s cappata al massimo (1000 Hz):
    #   dx = 343 / (1000 * 4) = 0.08575 m
    #   Griglia 10x8x4m -> 117 x 94 x 47 = ~516k voxels -> ~0.019 GB
    # Budget artificiale: 0.01 GB (10 MB) -> deve scattare MemoryConstraintError
    dense_profile = RoomAcousticProfile(
        length=10.0,
        width=8.0,
        height=4.0,
        alpha_per_surface={s: {125: 0.01} for s in SURFACE_KEYS},
        t60_target=2.5,
    )
    # Forza f_s al cap invocando direttamente con ppw=4
    # e simulando la stanza piccola con f_s alta tramite override del profilo
    # (in produzione il cap è garantito da get_schroeder_frequency).
    # Per il test usiamo ram_guardrail direttamente con dx piccolo forzato.
    dx_forced = C_AIR / (F_SCHROEDER_CAP * PPW)   # 0.08575 m
    try:
        PhysicsTranslator.ram_guardrail(
            dx=dx_forced,
            length=10.0,
            width=8.0,
            height=4.0,
            max_ram_gb=0.010,   # budget 10 MB — insufficiente per 516k voxels
        )
    except MemoryConstraintError as e:
        print(f"\n  [OK] MemoryConstraintError catturata correttamente:")
        print(f"  {e}")
        print(f"  required_gb  = {e.required_gb:.6f} GB")
        print(f"  max_gb       = {e.max_gb:.6f} GB")
        print(f"  grid_shape   = {e.grid_shape}")

    # ------------------------------------------------------------------ #
    # Test limite Schroeder — stanza piccola e riflettente                 #
    # ------------------------------------------------------------------ #
    print("\n--- Test Schroeder cap (atteso: WARNING + cap a 1000 Hz) ---")
    small_profile = RoomAcousticProfile(
        length=2.0,
        width=1.5,
        height=1.2,
        alpha_per_surface={s: {125: 0.01} for s in SURFACE_KEYS},
        t60_target=None,
    )
    f_s_capped = PhysicsTranslator.get_schroeder_frequency(small_profile)
    print(f"  f_s risultante: {f_s_capped:.2f} Hz (limite: {F_SCHROEDER_CAP:.0f} Hz)")

    print("\n" + "=" * 60)
    print("Self test completato.")
    print("=" * 60)