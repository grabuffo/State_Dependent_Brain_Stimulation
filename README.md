# State_Dependent_Brain_Stimulation
This repository includes the code required to reproduce the results in: "Targeting pre-stimulus brain states predicts and controls variability in stimulation responses - Rabuffo, G.; Angiolelli, M.; Fukai, T.; Deco, G.; Sorrentino, P.; Momi, D."

![alt text]([https://github.com/grabuffo/State_Dependent_Brain_Stimulation/blob/main/Rabuffo_et_al_Abstract.png](https://github.com/grabuffo/State_Dependent_Brain_Stimulation/blob/main/Rabuffo_et_al_Abstract.png))

# Data
The data used in this study were taken from an open dataset collected at the "Claudio Munari'' Epilepsy Surgery Center of Milan in Italy (https://doi.org/10.17605/OSF.IO/WSGZP), where simultaneous stereotactic electroencephalography (sEEG) and high-density scalp EEG (hd-EEG) was recorded following intracortical single pulse electrical stimulation on 36 patients (median age = 33 ± 8 years, 21 female). All subjects had a history of drug-resistant, focal epilepsy, and were candidates for surgical removal/ablation of the seizure onset zone (SOZ). For details regarding the data acquisition and the preprocessing steps please refer to the original papers (Mikulan et al., 2020, Parmigiani et al., 2022). In addition, it includes the spatial locations of the stimulating contacts in native MRI space, MNI152 space and Freesurfer's surface space, as well as the digitized positions of the 185 scalp hd-EEG electrodes.

# Notebooks
This repository contains Jupyter notebooks that replicate the code and results presented in our paper. 
Each notebook serves a specific purpose as described below:

| functions                         | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| functions                | This code provides a framework for analyzing neural data before and after a stimulus, focusing on channels near the stimulation site. The function channels_around_stimulus identifies which channels fall within a given radius of the stimulus, while MetricsOfInterest extracts different features (signal statistics, connectivity, network measures, and information-theoretic metrics) from the data, optionally restricted to those nearby channels. The main function analyze_pre_VS_post then splits the data into pre- and post-stimulus time windows, computes the metrics for each period, and compares them by calculating Spearman correlations and p-values between pre- and post-stimulus metrics. This allows us to quantify how neural activity changes in response to stimulation, both locally (within a radius) and across the entire network.
| surrogates      | This function generates a null dataset by trial-shuffling the channels independently within each session. This procedure preserves the temporal structure of each individual channel and maintains its trial-averaged response, but disrupts the cross-channel synchrony within single trials. As a result, the surrogate dataset retains identical marginal distributions and within-channel signal properties while eliminating trial-level dependencies that could reflect stimulation-induced effects. This shuffled dataset can then be used as a reference distribution to assess the statistical significance of observed correlations between metrics of interest (MOIs) in pre- and post-stimulus periods, allowing us to distinguish genuine stimulation effects from chance fluctuations.

## How to use the functions

```python
percentages = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ini_pre, fini_pre = 25, 250
ini_post, fini_post = 310, 900

output_dir = os.path.join(base_path, sub2use, "source_EEG_regions")
os.makedirs(output_dir, exist_ok=True)

for percentage in percentages:
    ch_index2use = np.argsort(distances)[:int(distances.shape[0] * percentage / 100)] 
    R = int(distances[ch_index2use[-1]]) + 1
    metrics_pre, metrics_post, Corr, Piva = fun.analyze_pre_VS_post(
        region_signals_per_trial[:,:-1,:], distances=distances, R=R,
        ini_pre=ini_pre, fini_pre=fini_pre,
        ini_post=ini_post, fini_post=fini_post
    )
    data_save = {
        'metrics_pre': metrics_pre,
        'metrics_post': metrics_post,
        'Corr': Corr,
        'Piva': Piva
    }
    output_path = os.path.join(base_path, sub2use, f"{sub2use}_{run}_{percentage}_metrics.pkl")
    with open(output_path, 'wb') as file:
        pickle.dump(data_save, file)
    print("Analysis complete and results saved at:", output_path)
```
## Example

| Notebook         | Run                 | Description                                                                 |
|-------------------|----------------|-------------------------------------------------------|
| extract_metrics               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/drive/1VeolR4xuSSancsqd3KYTuefPh4IGB700?usp=sharing]) | This notebook shows how to extract features from data
