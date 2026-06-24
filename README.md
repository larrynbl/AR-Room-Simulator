# AR-Room-Simulator

Feasibility study of CNN-based per-octave-band blind T30 reverberation time estimation from reverberant speech, targeting XR spatial audio applications.

**MSc Capstone Project** — Music and Acoustic Engineering, Politecnico di Milano  
Supervised by Prof. Lorenzo Picinali

---

## Overview

This project investigates whether a convolutional neural network can estimate T30 reverberation time across three octave bands (1 kHz, 2 kHz, 4 kHz) from reverberant speech Mel-spectrograms, without knowledge of the room geometry or source signal. The system uses a procedurally generated synthetic dataset of 1,486 room impulse responses across 9 room typologies, convolved with speech segments selected for sufficient natural pauses (VAD filtering).

The study characterises the sim-to-real gap arising from smartphone non-linear audio preprocessing (AGC), and demonstrates that per-band estimation is necessary for rooms with non-flat spectral decay profiles, which represent the majority of the dataset.

---

## Repository Structure

```
AR-Room-Simulator/
├── input_csv/                  # Room variant parameters (10 typologies × 150 variants)
│   ├── aula_variants_150.xlsx
│   ├── bagno_variants_150.xlsx
│   └── ...
├── data/
│   ├── dry_voices/             # LibriSpeech dev-clean + preprocessed WAV files
│   │   └── LibriSpeech/
│   └── materials/              # Absorption coefficient database (XLS/XLSX)
├── scripts/
│   ├── pyroom_generator.py     # Stage 1: RIR generation via pyroomacoustics ISM
│   ├── dsp_octave_t30.py       # Stage 2: Per-band T30 extraction (Schroeder integral)
│   ├── convolutore.py          # Stage 3: VAD-filtered speech convolution
│   ├── pulisci_dataset.py      # Stage 4: Dataset cleaning and filtering
│   ├── estrai_feature.py       # Stage 5: Mel-spectrogram extraction + Z-score norm
│   ├── train_cnn.py            # Stage 6: CNN training (asymmetric kernels)
│   ├── test_rumore.py          # Experiment 2: Pink noise robustness ablation
│   ├── app_ar_backend.py       # OSC integration with BeRTA binaural renderer
│   └── test_reale/
│       └── test_reale.py       # Experiment 3: Sim-to-real evaluation
├── dataset_ml_clean.csv        # Final ML dataset (844 samples, 9 room types)
├── master_log_t30.csv          # Full RIR log (1,486 entries across 10 typologies)
├── master_log_wet.csv          # Wet audio log (post-convolution)
├── risultati_test_XR.png       # Experiment 1: Scatter plot predicted vs ground truth
├── degrado_rumore_XR.png       # Experiment 2: MAE vs SNR across frequency bands
└── data/real_recs/             # Real smartphone recordings for sim-to-real test
```

---

## Pipeline (6 Stages)

**Stage 1 — RIR Generation** (`pyroom_generator.py`)  
Generates room impulse responses using pyroomacoustics 0.10.x (Image Source Method + Ray Tracing hybrid). 10 room archetypes × 150 parametric variants each, totalling 1,500 simulated rooms. Geometry, source/microphone positions, and per-surface absorption coefficients (125 Hz–4 kHz) are read from the `input_csv/` XLSX files. chiesa typology was fully excluded from the ML dataset as all variants exceed the T30 broadband filter threshold (>5s).

```bash
python scripts/pyroom_generator.py
```

**Stage 2 — T30 Extraction**  
T30 is extracted per octave band (1 kHz, 2 kHz, 4 kHz) and as a broadband value using the Schroeder backwards integration method, implemented in `dsp_octave_t30.py`. Output: `master_log_t30.csv`.

**Stage 3 — VAD-Filtered Convolution** (`convolutore.py`)  
Each RIR is convolved with a LibriSpeech speech segment selected to have ≥30% natural pauses (silence ratio measured on the dry signal). Speech segments that are too dense are cached and skipped. Output: `dataset_wet/` (1,487 WAV files) and `master_log_wet.csv`.

```bash
python scripts/convolutore.py
```

**Stage 4 — Dataset Cleaning** (`pulisci_dataset.py`)  
Filters on `t30_broadband ≤ 5.0s`. Final dataset: 844 samples across 9 room typologies.

```bash
python scripts/pulisci_dataset.py
```

**Stage 5 — Feature Extraction** (`estrai_feature.py`)  
Loads each wet WAV at 16 kHz, trims leading silence, computes a 128-band Mel-spectrogram (n_fft=2048, hop_length=256), converts to dB, applies per-image Z-score normalisation. Output: `dataset_spectrograms/` (844 × `.npy` files, shape 128×variable).

```bash
python scripts/estrai_feature.py
```

**Stage 6 — CNN Training** (`train_cnn.py`)  
Trains an asymmetric-kernel CNN (kernels: 3×7, 3×5, 3×3) on 675 training samples (80/20 split, random_state=42). Input shape: (128, 512, 1). Output: 3 linear nodes predicting T30 at 1 kHz, 2 kHz, 4 kHz. Loss: Huber. Optimizer: Adam. Early stopping patience: 15 epochs.

```bash
python scripts/train_cnn.py
```

---

## Pretrained Model

The trained model (`modello_xr_t30.h5`, ~127 MB) exceeds GitHub's file size limit and is stored externally.

📥 **Download:** [Google Drive — modello_xr_t30.h5](https://drive.google.com/file/d/1heE-K3qpUtnaiR_D6Ruv7g5uphKUc7Zj/view?usp=sharing)
Place the downloaded file in the repository root before running inference scripts.

---

## Results Summary

| Metric | 1 kHz | 2 kHz | 4 kHz |
|--------|-------|-------|-------|
| MAE (clean, synthetic) | 0.455 s | 0.362 s | 0.320 s |
| MAE (10 dB SNR pink noise) | ~0.9 s | — | — |
| Sim-to-real (smartphone) | severe degradation | frequency hierarchy violated | — |

The frequency hierarchy inversion observed on real smartphone recordings (estimated T30_4kHz > T30_1kHz) constitutes indirect evidence of non-linear signal processing in the acquisition chain (AGC), since this ordering is physically impossible in the ISM-simulated training distribution.

---

## Dependencies

```
python >= 3.9
tensorflow >= 2.11
keras
librosa
numpy
pandas
scipy
soundfile
pyroomacoustics == 0.10.x
scikit-learn
matplotlib
openpyxl
python-osc       # for app_ar_backend.py only
```

Install in a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

---

## Known Limitations

- **chiesa** typology excluded from ML dataset (T30 broadband > 5s in all 146 variants)
- **ristorante** underrepresented in ML dataset (16 samples after filtering)
- VAD filtering operates on the dry signal — in deployment, VAD must operate on the reverberant signal, which may not preserve silence ratios at high T30 values (>3.5s)
- System trained exclusively on ISM-simulated (shoebox) rooms; real-room generalisation is limited by the sim-to-real gap
- No frequency bands below 1 kHz estimated; low-frequency modal behaviour is outside the ISM validity range
- Pretrained model file not included in repository due to size constraints

---

## Citation

If you use this dataset or pipeline, please cite:

```
Salvatore Panta, "Blind Per-Band T30 Estimation from Reverberant Speech 
for XR Spatial Audio: A CNN Feasibility Study," 
MSc Capstone Report, Politecnico di Milano, 2026.
```

---

## License

Academic use only. LibriSpeech data is used under its original CC BY 4.0 license.
