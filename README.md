# State_Dependent_Brain_Stimulation
This repository includes the code required to reproduce the results in: "Pre-stimulus Brain States Predict and Control Variability in Stimulation Responses - G. Rabuffo; M. Angiolelli; T. Fukai; G. Deco; P. Sorrentino; D. Momi"

![alt text](https://github.com/grabuffo/State_Dependent_Brain_Stimulation/blob/main/Rabuffo_et_al_Abstract.png)

# Data
The data used in this study were taken from an open dataset collected at the "Claudio Munari" Epilepsy Surgery Center of Milan in Italy (https://doi.org/10.17605/OSF.IO/WSGZP), where simultaneous stereotactic electroencephalography (SEEG) and high-density scalp EEG (hd-EEG) was recorded following intracortical single pulse electrical stimulation on 36 patients (median age = 33 ± 8 years, 21 female). All subjects had a history of drug-resistant, focal epilepsy, and were candidates for surgical removal/ablation of the seizure onset zone (SOZ). For details regarding the data acquisition and preprocessing steps please refer to the original papers (Mikulan et al., 2020; Parmigiani et al., 2022). The dataset also includes the spatial locations of the stimulating contacts in native MRI space, MNI152 space, and Freesurfer's surface space, as well as the digitized positions of the 185 scalp hd-EEG electrodes.

# Repository Structure

```
src/                    # Core analysis functions
notebooks/              # Jupyter notebooks reproducing all figures
data/                   # Data files (large files excluded via .gitignore)
results/figures/        # Output figures
```

# Source Code

| File | Description |
|------|-------------|
| `src/functions.py` | Core functions for extracting metrics of interest (MOIs) from pre- and post-stimulus windows, including signal statistics, connectivity, network measures, and information-theoretic metrics. |
| `src/surrogates.py` | Generates null datasets by trial-shuffling channels independently within each session, preserving temporal structure while disrupting cross-channel trial-specific dependencies. |
| `src/functions_figs.py` | Plotting utilities used across figure notebooks. |

# Notebooks

## Data preparation

| Notebook | Description |
|----------|-------------|
| `extract_metrics.ipynb` | Extracts MOIs from raw epoched data across spatial radii for all subjects and sessions. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VeolR4xuSSancsqd3KYTuefPh4IGB700?usp=sharing) |
| `make_correlation_df_5MOIs.ipynb` | Builds the pre/post correlation dataframe for the 5 main metrics of interest across all sessions and radii. |
| `make_correlation_df_allMOIs_100radius.ipynb` | Same as above but for all available metrics at 100% radius, used for the extended correlation matrix (Supplementary Fig. S1). |

## Main figures

| Notebook | Description |
|----------|-------------|
| `Figure_1_A-B.ipynb` | Pre- vs. post-stimulus metric scatterplots and NRC curves (SEEG and hd-EEG). |
| `Figure_1_C.ipynb` | Session-level summary of pre/post correlations across participants. |
| `Figure_1_D.ipynb` | Out-of-sample prediction performance. |
| `Figure_2.ipynb` | Cross-metric correlation matrices and generalization frequency maps. |
| `Figure_3.ipynb` | Trial-to-trial variability before and after stimulation. |
| `Figure_4.ipynb` | Prospective closed-loop stimulation: state-conditioned triggering and variability reduction. |
| `Figure_5_radius_dependence_SEEG.ipynb` | Radius dependence of pre/post explained variance — SEEG. |
| `Figure_5_radius_dependence_hdEEG.ipynb` | Radius dependence of pre/post explained variance — hd-EEG. |
| `Figure_6_network_dependence_SEEG.ipynb` | Network-dependent predictability across stimulated functional networks — SEEG. |
| `Figure_6_network_dependence_hdEEG.ipynb` | Network-dependent predictability across stimulated functional networks — hd-EEG. |

## Supplementary figures

| Notebook | Description |
|----------|-------------|
| `Suppl_Figure_2.ipynb` | Surrogate test validating pre/post NRC correlations against channel-shuffle null. |
| `Suppl_Figure_3_Carryover_control_5MOIs_SEEG.ipynb` | Carryover control analysis (lagged predictor) — SEEG. |
| `Suppl_Figure_3_Carryover_control_5MOIs_hdEEG.ipynb` | Carryover control analysis (lagged predictor) — hd-EEG. |
| `Suppl_Figure_4_circular_trial_shift_5MOIs_SEEG.ipynb` | Circular trial-shift null model — SEEG. |
| `Suppl_Figure_4_circular_trial_shift_5MOIs_hdEEG.ipynb` | Circular trial-shift null model — hd-EEG. |
