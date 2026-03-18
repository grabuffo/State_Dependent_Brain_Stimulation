import numpy as np

# SHUFFLING TRIALS

def generate_null_data(data):
    channels, time_points, trials = data.shape
    null_data = np.empty_like(data)
    for i in range(channels):
        random_trials = np.random.permutation(trials)
        for j in range(trials):
            null_data[i, :, j] = data[i, :, random_trials[j]]
    return null_data

def phase_surrogate(ts, keep_first_ft=True, preserve_cross_spectrum=True):          
    offset = 1 if keep_first_ft else 0
    if ts.ndim == 1:
        ts = ts[:,np.newaxis]
    elif ts.ndim>2:
        raise ValueError('Wrong shape of input array: {ts.shape}')
    N,M = ts.shape
    N_ft = int(N/2 + 1) if N%2==0 else int((N+1)/2)
    if preserve_cross_spectrum:
        phi = np.random.uniform(high=2*np.pi, size=N_ft-offset)
    ts_surr = np.empty((N,M))
    for i in range(M):
        ft = np.fft.rfft(ts[:,i])
        ft_surr = ft[:]
        if not preserve_cross_spectrum:
            phi = np.random.uniform(high=2*np.pi, size=N_ft-offset)
        ft_surr[offset:] = ft_surr[offset:]*np.exp(1j*phi) 
        ts_surr[:,i] = np.fft.irfft(ft_surr,n=2*N_ft-2)
    return ts_surr.squeeze()