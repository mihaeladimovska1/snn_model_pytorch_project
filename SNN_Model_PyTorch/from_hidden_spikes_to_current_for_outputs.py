import numpy as np
import pickle

def from_spikes_to_current(hidden_spike_trains, weights):
    decay = 0.9802
    decays = 0.9231
    M=1000
    syn1 = np.zeros((hidden_spike_trains.shape[0]))
    syn1s = np.zeros((hidden_spike_trains.shape[0]))
    syn = np.zeros((hidden_spike_trains.shape[0],M))

    for i in range(hidden_spike_trains.shape[0]):
        for t in range(M):
            syn1[i] *= decay
            syn1s[i] *= decays
            if (hidden_spike_trains[i,t] == 1):
                syn1[i] += 1.0
                syn1s[i] += 1.0
            syn[i,t] = syn1[i] - syn1s[i]
    return np.matmul(np.transpose(weights[:,:10]),syn)
