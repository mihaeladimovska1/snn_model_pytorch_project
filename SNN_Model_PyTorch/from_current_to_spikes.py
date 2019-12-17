import numpy as np

def convert_current_to_spikes(i_in):
    #i_in is input current
    T = 0.1 # duration of simulation
    dt = 0.1e-3 #time step
    M = int(T//dt)
    tref = 3e-3 #refractory period
    MAX_spks=int(T//tref) #max no.of  spikes possible in the sim time
    t = np.arange(0, T, dt)

    #Neuron parameters
    capacity = 300e-12
    gL = 30E-9
    EL = -70e-3
    VT = 20e-3
    tau = 5e-3
    N = i_in.shape[0] #number of neurons
    w = 1.012e-10
    Ic = 2.7e-9 #min current needed for the neuron to spike for constant input
    Vm = np.zeros([N, M+1])
    #print(Vm.shape) = (256,1000)
    Y_spk = np.zeros([N, M])
    isref_n = np.zeros([N, 1])
    #now, get the spikes of the output neurons
    for i in range(M):
        #discretizing the CT DE
        k1 = (1/capacity)*(i_in[:,i]-(gL*(Vm[:,i]-EL)))
        k2 = (1/capacity)*(i_in[:,i] - (gL*(Vm[:,i] + k1*dt - EL)))
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
    return Y_spk

