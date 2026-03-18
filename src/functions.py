import numpy as np
from scipy import stats
from scipy.signal import butter, filtfilt, hilbert
from scipy.linalg import eig
from scipy.stats import skew, kurtosis, spearmanr, pearsonr, zscore
from lempel_ziv_complexity import lempel_ziv_complexity
from sklearn.decomposition import PCA
import networkx as nx
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
import bct
import pywt
from numba import njit
import antropy as ent
#from statsmodels.tsa.stattools import grangercausalitytests

@njit
def channels_around_stimulus(N, distances=None, R=None):
    #Parameters:
    #N number of channels
    #distances (ndarray): The distances of the channels from the stimulus(optional).
    #R (float): The radius within which to consider channels (optional).
    
    if distances is not None and R is not None:
        nregions_in_R=len(np.where(distances < R)[0]) # numbers of regions in radius R from the stimulus
        if nregions_in_R>1:
            channels_within_radius = np.where(distances < R)[0]
            channels_outside_radius = np.where(distances >= R)[0]
        else:
            print('Error: you need at least 2 channels in radius R from the stimulus')
    else:
        #print(f"Warning: You are calculating the measures on the entire network.")
        channels_within_radius = np.arange(N)
        channels_outside_radius = None
    return channels_within_radius, channels_outside_radius

def analyze_pre_VS_post(data, distances=None, R=None, ini_pre=None, fini_pre=None, ini_post=None, fini_post=None):
    #Analyze all data, pre and post-stimulus metrics. If distances and R are provided, only consider regions within the radius R from the stimulus target.
    
    #Parameters:
    #data (ndarray): The input data with shape (channels, time_points, trials).
    #distances (ndarray): The distances of the channels from the stimulus(optional).
    #R (float): The radius within which to consider channels (optional).
    #ini_pre (int): Initial time point for pre-stimulus period.
    #fini_pre (int): Final time point for pre-stimulus period.
    #ini_post (int): Initial time point for post-stimulus period.
    #fini_post (int): Final time point for post-stimulus period.
    
    #Returns:
    #metrics_pre (dict): Metrics for pre-stimulus period.
    #metrics_post (dict): Metrics for post-stimulus period.
    #Corr (ndarray): Correlation matrix between pre and post-stimulus metrics.
    #Piva (ndarray): P-value matrix for the correlations.

    # Errors and warnings:
    # Check if ini_pre, fini_pre, ini_post, fini_post are all provided

    channels, time_points, trials = data.shape

    if ini_pre is None or fini_pre is None or ini_post is None or fini_post is None:
        raise ValueError("Error: ini_pre, fini_pre, ini_post, and fini_post must all be provided.")

    # Check if ini_pre, fini_pre, ini_post, fini_post are valid
    max_index = len(data[0, :, 0])
    for index in [ini_pre, fini_pre, ini_post, fini_post]:
        if not isinstance(index, int) or index < 0 or index >= max_index:
            raise ValueError(f"Error: The input time indices must be integers, non-negative, and less than {max_index}.")

    # Check if ini_pre > fini_pre > ini_post > fini_post
    if not (0 <= ini_pre < fini_pre < ini_post < fini_post <= time_points):
        raise ValueError("Error: The indices must satisfy 0 <= ini_pre < fini_pre < ini_post < fini_post<= time_points).")

    # Check if distances and R are not provided
    if distances is None and R is None:
        N = len(data[:, 0, 0])
        print(f"Warning: You are calculating the measures on the entire network of N={N} channels.")
    
    ###########################################
    
    # Calculate pre-stimulus metrics on the channels in radius R from stimulation
    data_pre = data[:, ini_pre:fini_pre, :]
    metrics_pre = MetricsOfInterest(data_pre,distances=distances,R=R)

    # Calculate post-stimulus metrics on all the channels
    data_post = data[:, ini_post:fini_post, :]
    metrics_post = MetricsOfInterest(data_post,distances=None,R=None)

    metric_keys = list(metrics_pre.keys())

    Corr = np.zeros((len(metric_keys), len(metric_keys)))
    Piva = np.zeros((len(metric_keys), len(metric_keys)))

    for ii, key_pre in enumerate(metric_keys):
        for ij, key_post in enumerate(metric_keys):
            pre_values = np.array(list(metrics_pre[key_pre].values()))
            #print(pre_values)
            #print(key_pre)
            post_values = np.array(list(metrics_post[key_post].values()))
            #print(post_values)
            #print(key_post)
            correlation, p_value = stats.spearmanr(pre_values, post_values)
            Corr[ii, ij] = correlation
            Piva[ii, ij] = p_value
    return metrics_pre, metrics_post, Corr, Piva

def MetricsOfInterest(data, distances=None, R=None):
    
    #Parameters:
    #data (ndarray): The input data with shape (channels, time_points, trials).
    #distances (ndarray): The distances of the channels from the stimulus(optional).
    #R (float): The radius within which to consider channels (optional).
        
    channels, time_points, trials = data.shape
    channels_within_radius, channels_outside_radius = channels_around_stimulus(N=channels, distances=distances, R=R)
    data_in_R = data[channels_within_radius, :, :]
    
    Concatenated_trials_in_R = np.reshape(data_in_R, (len(channels_within_radius), -1))
    ZConcatenated_trials_in_R = stats.zscore(Concatenated_trials_in_R, axis=1)
    Zdata_in_R = np.reshape(ZConcatenated_trials_in_R, (len(channels_within_radius), time_points, trials))

    metrics = {i: {} for i in range(trials)}

    feature_extractors = {
        'signals_statistics': lambda data, Zdata: extract_signals_statistics(data, Zdata),
        'connectivity_statistics': lambda matrix, distances=None, R=None: extract_connectivity_statistics(matrix, distances=distances, R=R),
        'network_measures': lambda matrix, distances=None, R=None: extract_network_measures(matrix, distances=distances, R=R),
        'information_and_complexity': lambda data: extract_info_and_complexity_measures(data)
    }
    
    print('MetricsOfInterest loop through trials...')
    
    for i in range(trials):
        print(f"compute features for trial {i}")
        # trial data
        dat = data[:, :, i].T

        # trial data with channels in radius R from stimulation
        dat_in_R = data_in_R[:, :, i].T

        # Trial in concatenated z-scored data
        Zdat_in_R = Zdata_in_R[:, :, i].T

        # Measure connectivity matrices from the entire network
        connectivity_matrices = extract_connectivity_matrices(dat)

        # Measure signal statistics of data in R
        signal_stats = feature_extractors['signals_statistics'](dat_in_R, Zdat_in_R)
        for key, value in signal_stats.items():
            metrics[i][key] = value

        # Measure connectivity statistics
        for matrix_type, matrix_data in connectivity_matrices.items():
            connstats_metrics = feature_extractors['connectivity_statistics'](matrix_data, distances=distances, R=R)
            for key, value in connstats_metrics.items():
                metrics[i][f"{matrix_type}_{key}"] = value

        # Measure network measures from connectivity matrices
        for matrix_type, matrix_data in connectivity_matrices.items():
            network_metrics = feature_extractors['network_measures'](matrix_data, distances=distances, R=R)
            for key, value in network_metrics.items():
                metrics[i][f"{matrix_type}_{key}"] = value

        # Measure information and complexity 
        info_complexity_metrics = feature_extractors['information_and_complexity'](dat_in_R)
        for key, value in info_complexity_metrics.items():
            metrics[i][key] = value
            
    return invert_dict(metrics)


###################################################

# Functions to extract measures belonging to 4 categories:
# 1) signals_statistics
# 2) connectivity_statistics
# 3) network_measures
# 4) info_and_complexity_measures

def extract_signals_statistics(data, Zdata):
    # data: time x channels    
    mean_data = np.mean(data)
    var_data = np.var(data)
    mean_var_data = np.mean(np.var(data, axis=0))
    var_mean_data = np.var(np.mean(data, axis=0))
    var_var_data = np.var(np.var(data, axis=0))
    mean_skewness_data = np.mean(skew(data, axis=0))
    var_skewness_data = np.var(skew(data, axis=0))
    mean_kurt_data = np.mean(kurtosis(data, axis=0))
    var_kurt_data = np.var(kurtosis(data, axis=0))
    
    mean_peak_to_peak = np.mean(np.ptp(data, axis=0))
    var_peak_to_peak = np.var(np.ptp(data, axis=0))
    zero_crossing_rate = np.mean(np.sum(np.diff(np.sign(data), axis=0) != 0, axis=0) / data.shape[0])
    mean_square = np.mean(np.square(data))

    salience = np.mean(np.abs(Zdata)) 
    NDP=np.mean(Zdata**2,axis=1)   
    max_dnp=np.max(NDP)
    var_dnp=np.var(NDP)
    return {
        'mean_data': mean_data,
        'var_data': var_data,
        'mean_var_data': mean_var_data,
        'var_mean_data': var_mean_data,
        'var_var_data': var_var_data,
        'mean_skewness_data': mean_skewness_data,
        'var_skewness_data': var_skewness_data,
        'mean_kurt_data': mean_kurt_data,
        'var_kurt_data': var_kurt_data,
        'mean_peak_to_peak': mean_peak_to_peak,
        'var_peak_to_peak': var_peak_to_peak,
        'zero_crossing_rate': zero_crossing_rate,
        'mean_square': mean_square,
        'salience': salience,
        'max_dynamic_network_profile': max_dnp,
        'var_dynamic_network_profile':var_dnp
    }
    
@njit 
def extract_connectivity_statistics(matrix, distances=None, R=None):
    
    #Parameters:
    #matrix (ndarray): The input data is a square connectivity matrix.
    #distances (ndarray): The distances of the channels from the stimulus(optional).
    #R (float): The radius within which to consider channels (optional).

    N=len(matrix)
    
    channels_within_radius, channels_outside_radius = channels_around_stimulus(N=N, distances=distances, R=R)

    # matrix of channels in radius R from stimulus
    n = len(channels_within_radius)
    reduced_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            reduced_matrix[i, j] = matrix[channels_within_radius[i], channels_within_radius[j]]
    
    reduced_matrix_tri = []
    for i in range(n):
        for j in range(i+1, n):
            reduced_matrix_tri.append(reduced_matrix[i, j])
    
    reduced_matrix_tri = np.array(reduced_matrix_tri)    # calculated on reduced vectorized matrix

    mean_matrix_tri = np.mean(reduced_matrix_tri)
    var_matrix_tri = np.var(reduced_matrix_tri)
    skw_matrix_tri = my_skew(reduced_matrix_tri) 
    kurt_matrix_tri = my_kurtosis(reduced_matrix_tri)
    median_matrix_tri = np.median(reduced_matrix_tri) #nonzero_elements = reduced_matrix[reduced_matrix != 0]   why this was here?
    mom1_matrix = mean_matrix_tri / np.std(reduced_matrix_tri) if np.std(reduced_matrix_tri) != 0 else np.nan
    CV_matrix = np.std(reduced_matrix_tri) / mean_matrix_tri if mean_matrix_tri != 0 else np.nan

    # calculated on reduce matrix
    norm_matrix = np.linalg.norm(reduced_matrix)
    #max_eigen_val_matrix = my_calculate_max_eigenvalue(reduced_matrix)

    # calculated on entire matrix 
    mean_matrix_all = np.zeros(N)
    var_matrix_all = np.zeros(N)
    for i in range(N):
        mean_matrix_all[i] = my_mean(matrix[:, i])
        var_matrix_all[i] = my_var(matrix[:, i])

    mean_mean_matrix = np.mean(mean_matrix_all[channels_within_radius])
    mean_var_matrix = np.mean(var_matrix_all[channels_within_radius])
    var_mean_matrix = np.var(mean_matrix_all[channels_within_radius])
    var_var_matrix = np.var(var_matrix_all[channels_within_radius])

    return {
        'mean_mat': mean_matrix_tri,
        'var_mat': var_matrix_tri,
        'median_mat': median_matrix_tri,
        'skewness_mat': skw_matrix_tri,
        'kurtosis_mat': kurt_matrix_tri,
        'first_moment_mat': mom1_matrix,
        'coefficient_of_variation_mat': CV_matrix,
        'norm': norm_matrix,
        'mean_mean_mat': mean_mean_matrix,
        'mean_var_mat': mean_var_matrix,
        'var_mean_mat': var_mean_matrix,
        'var_var_mat': var_var_matrix
        }
    
def extract_network_measures(matrix, distances=None, R=None):
    #Parameters:
    #matrix (ndarray): The input data is a square connectivity matrix.
    #distances (ndarray): The distances of the channels from the stimulus(optional).
    #R (float): The radius within which to consider channels (optional).
    
    channels_within_radius, channels_outside_radius = channels_around_stimulus(N=len(matrix), distances=distances, R=R)

    if len(channels_within_radius) == 0:
        raise ValueError("No channels found within the specified radius.")
    
    # matrix of channels in radius R from stimulus
    n = len(channels_within_radius)
    reduced_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            reduced_matrix[i, j] = matrix[channels_within_radius[i], channels_within_radius[j]]

    # calculated on reduced matrix
    G_red = nx.from_numpy_array(reduced_matrix)
    communities = list(nx.algorithms.community.label_propagation_communities(G_red))
    modularity = nx.algorithms.community.modularity(G_red, communities) 
    charpath_length=bct.charpath(reduced_matrix)[0]

    # calculated on the entire matrix and average within a radius
    G = nx.from_numpy_array(matrix) 

    clustering_coefficient = bct.clustering_coef_wu(matrix)
    avg_clustering_coefficient = np.mean(clustering_coefficient[channels_within_radius])

    eigenvector_centrality = np.asarray(list(nx.eigenvector_centrality_numpy(G, weight='weight').values())) 
    avg_eigenvector_centrality = np.mean(eigenvector_centrality[channels_within_radius])   

    return {
        'avg_clustering_coefficient': avg_clustering_coefficient,
        'avg_eigenvector_centrality': avg_eigenvector_centrality,
        'modularity': modularity,
        'charpath_length': charpath_length,
    }

def extract_info_and_complexity_measures(data):
    # data: time x channels       
    Dims = Dimensionality(data)
    LZ = LempelZiv(data)
    PCI = pci_lz_complexity(data)
    Metastab = metastability(data)
    
    spect_ratio = SpectralRatio(data)
    power_entropy = power_shannon_entropy(data)

    Entropy_pca=entropy_pca(data)

    #connectivity matrices
    fc, dfc, plv_matrix, aec, cokurto, pcm =list(extract_connectivity_matrices(data).values())    
    
    Entropy_FC = eigenvalues_shannon_entropy(fc)
    Entropy_DFC = eigenvalues_shannon_entropy(dfc)
    Entropy_PLV = eigenvalues_shannon_entropy(plv_matrix)
    Entropy_AEC =  eigenvalues_shannon_entropy(aec)
    Entropy_COKU = eigenvalues_shannon_entropy(cokurto)
    Entropy_PCM =  eigenvalues_shannon_entropy(pcm)
    
    return {
        'Dimensionality': Dims,
        'Lempel-Ziv': LZ,
        'PCI': PCI,
        'Metastability': Metastab,
        'Spectral Ratio': spect_ratio,
        'Power_Shannon_Entropy': power_entropy,
        'Entropy_PCA': Entropy_pca,
        'Entropy_FC': Entropy_FC,
        'Entropy_DFC': Entropy_DFC,
        'Entropy_PLV': Entropy_PLV,
        'Entropy_AEC': Entropy_AEC,
        'Entropy_COKU': Entropy_COKU,
        'Entropy_PCM': Entropy_PCM
    }



######################################################
# CONNECTIVITY MATRICES

def extract_connectivity_matrices(data):
    #input: signals data ~(time,channels)
    
    fc = np.corrcoef(data.T) # Functional connectivity (Pearson's correlation matrix)
    edg = go_edge(data)   # edge co-activations            
    dfc = np.corrcoef(edg) # time-resolved dynamic functional connectivity
    aec = computeAEC(data) 
    plv_matrix = compute_PLVmatrix(data) #phase locking value matrix
    cokurto = compute_cokurtosis(data) #cokurtosis matrix
    pcm=phase_coherence_matrix(data)
            
    return {
        'functional_connectivity_matrix': fc,
        'dynamic_functional_connectivity_matrix': dfc,
        'plv_matrix': plv_matrix,
        'aec_matrix': aec, 
        'co-kurtosis_matrix': cokurto,
        'phase_coherence_matrix':pcm
    }

def compute_cokurtosis(data):
    # Data shape: (time, nchannels)
    time, nchannels = data.shape
    
    # Compute means and centered data
    means = np.mean(data, axis=0)
    centered_data = data - means
    
    # Compute fourth moments and variances
    fourth_moments = np.mean(centered_data**4, axis=0)
    variances = np.var(data, axis=0)
    
    # Compute the outer products of centered data
    centered_data_sq = centered_data**2
    outer_products = np.einsum('ti,tj->ij', centered_data_sq, centered_data_sq) / time

    # Compute the cokurtosis matrix
    cokurtosis_matrix = outer_products - np.outer(variances, variances)
    
    # Diagonal elements need to be adjusted
    np.fill_diagonal(cokurtosis_matrix, fourth_moments - variances**2)
    
    return cokurtosis_matrix


def computeAEC(data):
    # Input: time x channels    
    # Compute the amplitude envelope for all channels
    envelopes = np.abs(hilbert(data, axis=0))
 
    # Compute the AEC matrix
    aec_matrix = np.corrcoef(envelopes, rowvar=False)
    
    return aec_matrix

@njit
def phase_locking_value(phase1, phase2):
    """
    Compute the Phase Locking Value (PLV) between two phase arrays.
    """
    # Compute the phase difference
    phase_diff = phase1 - phase2
    # Compute the PLV
    return np.abs(np.mean(np.exp(1j * phase_diff)))

def compute_PLVmatrix(data, fs=1000, lowcut=13, highcut=30, order=4):
    num_channels = data.shape[1]
    
    # Bandpass Filter
    b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
    filtered_data = np.apply_along_axis(lambda x: filtfilt(b, a, x), 0, data)
    
    # Compute Analytical Signals and Phases
    analytical_signals = hilbert(filtered_data, axis=0)
    phases = np.angle(analytical_signals)
    
    # Compute the PLV Matrix
    plv_matrix = np.zeros((num_channels, num_channels))
    
    for j in range(num_channels):
        for k in range(j + 1, num_channels):
            plv_matrix[j, k] = phase_locking_value(phases[:, j], phases[:, k])
    
    # Fill the symmetric part of the matrix
    plv_matrix = plv_matrix + plv_matrix.T
    
    return plv_matrix

def phase_coherence_matrix(data):
    """
    Compute the phase coherence matrix from time series data.
    """
    n_samples, n_channels = data.shape
    analytic_signal = hilbert(data, axis=0)                               # Compute the analytic signal using the Hilbert transform
    instantaneous_phase = np.angle(analytic_signal)                        # Extract the instantaneous phase
    coherence_matrix = np.zeros((n_channels, n_channels))                  # Initialize the coherence matrix

    for i in range(n_channels):
        for j in range(i, n_channels):
            phase_diff = instantaneous_phase[:, i] - instantaneous_phase[:, j]
            coherence = np.abs(np.sum(np.exp(1j * phase_diff)) / n_samples)
            coherence_matrix[i, j] = coherence
            coherence_matrix[j, i] = coherence  # Symmetric matrix
    
    return coherence_matrix

###########################################################
###########################################################
###########################################################
# SUPPORT FUNCTIONS

def invert_dict(metrics):
    """
    Inverts a nested dictionary such that for each subkey, a new dictionary
    is created containing the original keys and their corresponding values.

    Parameters:
    metrics (dict): A dictionary where each value is a dictionary with a common set of keys.

    Returns:
    dict: A new dictionary where each key is a subkey from the original dictionaries,
          and each value is a dictionary mapping original keys to their values.
    """
    inverted_dict = {}

    # Iterate over the original dictionary
    for outer_key, inner_dict in metrics.items():
        for subkey, value in inner_dict.items():
            if subkey not in inverted_dict:
                # Initialize a new dictionary for this subkey
                inverted_dict[subkey] = {}
            # Add the value to the new dictionary
            inverted_dict[subkey][outer_key] = value

    return inverted_dict

def bin_seq(arr):
    return ''.join(str(i) for i in arr.flatten())

@njit
def my_mean(arr):
    return np.sum(arr) / arr.size

@njit
def my_var(arr):
    mean = my_mean(arr)
    return np.sum((arr - mean) ** 2) / arr.size

@njit
def my_skew(data):
    n = len(data)
    mean = np.mean(data)
    m3 = np.mean((data - mean)**3)
    m2 = np.std(data)
    if m2 == 0:
        return 0
    else:
        return (n * m3) / ((n - 1) * (n - 2) * (m2**3))
@njit
def my_kurtosis(data):
    n = len(data)
    mean = np.mean(data)
    m4 = np.mean((data - mean)**4)
    m2 = np.var(data)
    if m2 == 0:
        return 0
    else:
        return (n * (n + 1) * m4) / ((n - 1) * (n - 2) * (n - 3) * (m2**2)) - (3 * (n - 1)**2) / ((n - 2) * (n - 3))

@njit
def kuramoto_order_parameter(phases):
    return np.abs(np.sum(np.exp(1j * phases), axis=0)) / phases.shape[0]

def go_edge(tseries):
    nregions=tseries.shape[1]
    iTriup= np.triu_indices(nregions,k=1) 
    gz=stats.zscore(tseries)
    Eseries = gz[:,iTriup[0]]*gz[:,iTriup[1]]
    return Eseries
    
def metastability(data):
    analytic_signal = hilbert(data, axis=0)
    instantaneous_phase = np.angle(analytic_signal)
    ko_parameter = kuramoto_order_parameter(instantaneous_phase)
    return np.var(ko_parameter)

def Dimensionality(data):
    # Manifold dimensionality: PCA complexity
    n_samples, n_features = data.shape
    n_components = min(n_samples, n_features)
    pca = PCA(n_components=n_components)
    pca.fit(data)
    sumv = 0.
    j = 0
    while sumv < 0.95 and j < len(pca.explained_variance_ratio_):  #not 0.99 and I put a control on n_components which has be less of n_samples
        sumv += pca.explained_variance_ratio_[j]
        j += 1  
    return j

def entropy_pca(data):
    # Manifold dimensionality: PCA complexity
    n_samples, n_features = data.shape
    n_components = min(n_samples, n_features)
    pca = PCA(n_components=n_components)
    pca.fit(data)
    p=pca.explained_variance_ratio_  #not 0.99 and I put a control on n_components which has be less of n_samples
    epsilon = 1e-12                                         # Step 3: Calculate the Shannon entropy on the distribution of eigenvalues
    p = np.maximum(p, epsilon)
    ent_pca = -np.sum(p * np.log(p))
    
    return ent_pca

def SpectralRatio(data):
    # Calculate spectral ratio
    n_samples, n_features = data.shape
    n_components = min(n_samples, n_features)
    pca = PCA(n_components=n_components)
    pca.fit(data)
    eigenvalues = pca.explained_variance_
    largest_eigenvalue = eigenvalues[0]
    spectral_ratio = largest_eigenvalue / eigenvalues.sum()
    return spectral_ratio

def power_shannon_entropy(data):
    n_samples, n_channels = data.shape
    pse_channels = ent.spectral_entropy(data, sf=1000, axis=0, nperseg=128, method='welch', normalize=True)
    # Average over all channels (ROIs)
    pse_mean = np.mean(pse_channels)
    return pse_mean

@njit
def eigenvalues_shannon_entropy(connectivity_matrix):
    eigenvalues = np.linalg.eigvalsh(connectivity_matrix)   # Step 1: Compute the eigenvalues of the covariance matrix
    eigenvalues_sum = np.sum(eigenvalues)                   # Step 2: Normalize the eigenvalues
    p = eigenvalues / eigenvalues_sum
    epsilon = 1e-12                                         # Step 3: Calculate the Shannon entropy on the distribution of eigenvalues
    p = np.maximum(p, epsilon)
    entropy = -np.sum(p * np.log(p))
    return entropy

def my_calculate_max_eigenvalue(matrix):
    if matrix.shape[0] == matrix.shape[1]:
        eigen_vals_matrix, _ = np.linalg.eig(matrix)
        mm = np.max(eigen_vals_matrix)
        return np.exp(mm)
    else:
        return np.nan

# Threshold is to be adjusted
def LempelZiv(data, percentage=5): #data ~ (time,channels)
    time_points, channels = data.shape
    oneper=int(percentage*time_points*channels/100) # how many datapoints represent percentage% of dataset
    z=stats.zscore(data)
    th=np.sort(np.abs(z.flatten()))[-oneper] #threshold at percentage% of datapoints
    Zbin=np.where(np.abs(z)>th,1,0) 
    LZ=lempel_ziv_complexity(bin_seq(Zbin))  #LZ complexity
    return LZ
    
def pci_lz_complexity(data):
    perturbed_data = data + np.random.randn(*data.shape) * 0.01  # Adding small noise as perturbation
    pci = LempelZiv(perturbed_data)  # Calculate LZ complexity of perturbed data
    lz_data=LempelZiv(data)
    return pci/lz_data