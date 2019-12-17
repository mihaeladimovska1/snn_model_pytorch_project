# first map each pixel \in [0,255] to a spike train of len(N); N=1000
# then load the maps from the conv. layer and convert them to pixels

import numpy as np
import matplotlib.pyplot as plt
import pickle

T = 0.1 # duration of simulation

dt = 0.1e-3 #time step
M = int(T//dt)
tref = 3e-3 # refractory period
MAX_spks = int(T//tref) #max no.of  spikes possible in the sim time

t = np.arange(0, T, dt)
#t.shape = (1000,1)
#Neuron parameters
capacity = 300e-12
gL = 30E-9
EL = -70e-3
VT = 20e-3
tau = 5e-3
tau_s = tau / 4
tau_Ld = 1e-3
tau_N = 10e-3

# Input pixels:
N_pixel = 256
pix =[i for i in range(N_pixel)]
pix = np.asarray(pix)
pix = np.transpose(pix)
#print(pix.shape) = (256,1)

w = 1.012e-10
Ic = 2.7e-9 #min current needed for the neuron to spike for constant input
Vm = np.zeros([N_pixel, M+1])
#print(Vm.shape) = (256,1000)
Y_spk = np.zeros([N_pixel, M])
isref_n = np.zeros([N_pixel, 1])
i_in = Ic + pix * w
print(i_in.shape)
exit(1)
# i_in = 2700e-12 + pix * w;
for i in range(M):
    #discretizing the CT DE
    k1 = (1/capacity)*(i_in-(gL*(Vm[:,i]-EL)))
    k2 = (1/capacity)*(i_in - (gL*(Vm[:,i] + k1*dt - EL)))
    #print(k1)
    #print(k2)
    #exit(1)

    Vm[:,i+1] = Vm[:,i] + dt*(k1+k2)/2
    Vm[np.where(t[i] - isref_n < tref), i+1] = EL
    spind = []
    for j in range(Vm[:,i+1].shape[0]):
        if Vm[j,i+1]-VT > 0:
            spind.append(1)
        else:
            spind.append(-1)
    spind = np.asarray(spind)
    if max(spind)>0:
        #print("heree")
        resetfind_n = np.where(spind>0)
        isref_n[resetfind_n] = t[i]
        Vm[resetfind_n, i] = VT
        Y_spk[resetfind_n, i] = 1

#plot the frequency of the spikes for each pixel

#save the spike trains for every pixel

fw = open('project_data/pixel_spike_trains', 'wb')
pickle.dump(Y_spk, fw)
fw.close()

freq = [sum(Y_spk[i,:]) for i in range(256)]
print(np.transpose(np.asarray(freq)))
plt.plot([i for i in range(256)], freq)
plt.xlim([0,255])
plt.xticks([0,45,95,135,185,205,255])
plt.show()

#plot the spike trains of each pixel

[x,y] = np.where(Y_spk)
plt.plot(y,x,'o')
plt.show()