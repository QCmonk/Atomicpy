from capy import *
from operators import *
from qChain import *
from utility import *
from IPython import embed

import os
import time
import numpy as np 
from scipy import signal
import matplotlib.pyplot as plt
import cmath
import h5py

def getNormalModes(t,y): 
    #has a threshold normal modes of 1/100 of absolute maxima of fft
    FreqAmp = np.fft.fftshift(np.abs(np.fft.fft(y)))
    Freq = np.fft.fftshift(np.fft.fftfreq(len(y), t[1] - t[0]))
    #splice = int(np.floor(len(FreqAmp)/2))
    #FreqAmp = FreqAmp[:splice]
    #Freq = Freq[:splice]
    threshAmp = max(FreqAmp) / 10
    NormalModes = []
    for i in range (1,len(FreqAmp)-1):
        if ((FreqAmp[i]>FreqAmp[i-1]) & (FreqAmp[i]>FreqAmp[i+1]) & (FreqAmp[i]>=threshAmp)):
            NormalModes.append(Freq[i])
    return [Freq, FreqAmp, NormalModes]


def LPFilter(dataset, filter_frequency, sampling_rate):
    nyquist_rate = 0.5*sampling_rate
    fraction_nyquist = filter_frequency/nyquist_rate
    b,a = signal.butter(5, fraction_nyquist)
    #b, a = signal.cheby2(5, 40, fraction_nyquist)
    filtered_out = 2*signal.filtfilt(b, a, dataset, padlen=0)
    return filtered_out



def epsilon_find(sim_vars, erange):
    """
    Computes the reconstruction error using a range of epsilon values for 
    the same simulation parameters. 
    """
    errors = []
    for epsilon in erange:
        sim_vars["epsilon"] = epsilon
        original, recon = recon_pulse(sim_vars, plot=False, savefig=False)
        # compute error
        errors.append(rmse(original,recon))

    # plot RMSE error against epsilon
    plt.plot(erange, errors)
    plt.figure(num=1, size=[16,9])
    plt.xlabel("Epsilon")
    plt.ylabel("RMSE")
    plt.title("Reconstruction Error vs. Epsilon")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    ###############################################
    # Compressive Reconstruction of Neural Signal #
    ###############################################
    # set random seed for tuning frequency choice
    np.random.seed(141)
    # define user parameters for simulation run
    # sim_vars = {"measurements":        50,           # number of measurements
    #             "epsilon":           0.01,           # radius of hypersphere in measurement domain
    #             "sig_amp":             40,           # amplitude of magnetic field signal in Gauss/Hz
    #             "sig_freq":          5023,           # frequency of magnetic field signal
    #             "tau":              0.033,           # time events of pulses
    #             "f_range":  [4500,5500,2],            # frequency tunings of BECs
    #             "noise":             0.00,           # noise to add to measurement record SNR percentage e.g 0.1: SNR = 10
    #             "zlamp":                0,
    #             "method":       "default",
    #             "savef":             5000,
    #             "fs":               2.2e4}           # sasvempling rate for measurement transform

    # # define user parameters for simulation run
    # sim_vars = {"measurements":        40,           # number of measurements
    #             "epsilon":           0.01,           # radius of hypersphere in measurement domain
    #             "sig_amp":             10,           # amplitude of magnetic field signal in Gauss/Hz
    #             "sig_freq":          1000,           # frequency of magnetic field signal
    #             "tau":               0.01,           # time events of pulses
    #             "f_range":   [900,1100,1],            # frequency tunings of BECs
    #             "noise":             0.00,           # noise to add to measurement record SNR percentage e.g 0.1: SNR = 10
    #             "zlamp":                0,
    #             "zlfreq":            1000,
    #             "method":       "default",
    #             "savef":             5000,
    #             "fs":               2.2e4}           # sasvempling rate for measurement transform



    # epsilon range
    #erange = np.linspace(0.0,0.05, 50)
    #epsilon_find(sim_vars, erange)

    #bfreqs1, projs1 = recon_pulse(sim_vars, plot=True, savefig=True)
    
    # sim_vars["zlamp"] = 0
    # bfreqs2, projs2 = recon_pulse(sim_vars, plot=False) v 

    # # plot those bad boys against each other
    # #plt.style.use('dark_background')
    # plt.plot(bfreqs1, -(2*projs1-1), 'o-', alpha=0.3,linewidth=0.6)
    # # plt.plot(bfreqs2, projs2, 'g-')
    # plt.title(r" 'Rabi' spectroscopy for 1000 Hz $\sigma_z$ AC signal")
    # # plt.legend([r"With $\sigma_z $ line noise",r"Without $\sigma_z $ line noise"])
    # plt.ylabel(r"$\langle F_z \rangle $")
    # plt.xlabel("Rabi Frequency (Hz)")
    # plt.figure(num=1, figsize=[16,9])
    # plt.show()
    #exit()
    ###############################################
    


    # # set the bajillion simulation parameters. There simply isn't a better way to do this. 
    # # define generic Hamiltonian parameters with Zeeman splitting and rf dressing

    params = {"tstart":      0.0,              # time range to simulate over
              "tend":       10.0,
              "dt":         1e-8,
              "larmor":     7e5,              # bias frequency (Hz)
              "rabi":       104.0,              # dressing amplitude (Hz)
              "rff":        7e5 - 9e3,              # dressing frequency (Hz)
              "rph":       0.0,
              "quad":       9e3,             #quadratic Zeeman Shift (Hz)
              "tr2":        np.sqrt(0.5)*1/104, #time to start second Rabi frequency. Must be >= tstart np.sqrt(2)*
              "xlamp":       104.0,              # amplitude of 
              "xlfreq":     7e5 + 9e3,
              "xlphase":     0.0,
              "nf":         1e8,              # neural signal frequency (Hz)
              "sA":            0,              # neural signal amplitude (Hz/G)
              "nt":          5e2,              # neural signal time event (s)
              "dett":        60.0,              # detuning sweep start time
              "detA":          0,              # detuning amplitude
              "dete":       60.0,              # when to truncate detuning
              "beta":         0.0,              # detuning temporal scaling
              "zlamp":         0,
              "zlfreq":       50,
              "zlphase":     0.0,
              "proj": meas2["0"],              # measurement projector
              "savef":       int(1e6)}              # plot point frequency
 
    atom = SpinSystem(spin="one", init = '1')

    t, F, states  = atom.state_evolve(params=params, bloch=[False, int(1e5)])  
    atom.norm_plot()


    '''with h5py.File('10sec_sim_1.h5', 'w') as hf:
        hf.create_dataset("<F>",  data=F[:,0])     
        hf.create_dataset("rabi", data = params["rabi"])
        hf.create_dataset("larmor", data = params["larmor"]) 
        hf.create_dataset("quad", data = params["quad"])               
        hf.create_dataset("init state", data = [0,1,0])'''
    


    '''
    params["rabi"] = 95.0
    params["quad"] = 1.1e4
    params["rff"]  = params["larmor"] - params["quad"]
    params["xlfreq"]  = params["larmor"] + params["quad"]
    params["tr2"] = np.sqrt(0.5)*1/params["rabi"]
    params["xlamp"] = params["rabi"]

    with h5py.File('10sec_sim_2.h5', 'w') as hf:
        hf.create_dataset("<F>",  data=F[:,0])     
        hf.create_dataset("rabi", data = params["rabi"])
        hf.create_dataset("larmor", data = params["larmor"]) 
        hf.create_dataset("quad", data = params["quad"])               
        hf.create_dataset("init state", data = [0,1,0])



    params["rabi"] = 95.0
    params["quad"] = 1.2e4
    params["rff"]  = params["larmor"] - params["quad"]
    params["xlfreq"]  = params["larmor"] + params["quad"]
    params["tr2"] = np.sqrt(0.5)*1/params["rabi"]
    params["xlamp"] = params["rabi"]

    atom.reset()
    t, F, states  = atom.state_evolve(params=params, bloch=[False, int(1e5)])  


    with h5py.File('10sec_sim_3.h5', 'w') as hf:
        hf.create_dataset("<F>",  data=F[:,0])     
        hf.create_dataset("rabi", data = params["rabi"])
        hf.create_dataset("larmor", data = params["larmor"]) 
        hf.create_dataset("quad", data = params["quad"])               
        hf.create_dataset("init state", data = [0,1,0])


    params["rabi"] = 91.0
    params["quad"] = 0.9e4
    params["rff"]  = params["larmor"] - params["quad"]
    params["xlfreq"]  = params["larmor"] + params["quad"]
    params["tr2"] = np.sqrt(0.5)*1/params["rabi"]
    params["xlamp"] = params["rabi"]

    atom.reset()
    t, F, states  = atom.state_evolve(params=params, bloch=[False, int(1e5)])  


    with h5py.File('10sec_sim_4.h5', 'w') as hf:
        hf.create_dataset("<F>",  data=F[:,0])     
        hf.create_dataset("rabi", data = params["rabi"])
        hf.create_dataset("larmor", data = params["larmor"]) 
        hf.create_dataset("quad", data = params["quad"])               
        hf.create_dataset("init state", data = [0,1,0])


    params["rabi"] = 87.0
    params["quad"] = 0.9e4
    params["rff"]  = params["larmor"] - params["quad"]
    params["xlfreq"]  = params["larmor"] + params["quad"]
    params["tr2"] = np.sqrt(0.5)*1/params["rabi"]
    params["xlamp"] = params["rabi"]

    atom.reset()
    t, F, states  = atom.state_evolve(params=params, bloch=[False, int(1e5)])  


    with h5py.File('10sec_sim_5.h5', 'w') as hf:
        hf.create_dataset("<F>",  data=F[:,0])     
        hf.create_dataset("rabi", data = params["rabi"])
        hf.create_dataset("larmor", data = params["larmor"]) 
        hf.create_dataset("quad", data = params["quad"])               
        hf.create_dataset("init state", data = [0,1,0])


    params["rabi"] = 105.0
    params["quad"] = 0.9e4
    params["rff"]  = params["larmor"] - params["quad"]
    params["xlfreq"]  = params["larmor"] + params["quad"]
    params["tr2"] = np.sqrt(0.5)*1/params["rabi"]
    params["xlamp"] = params["rabi"]


    atom.reset()
    t, F, states  = atom.state_evolve(params=params, bloch=[False, int(1e5)])  


    with h5py.File('10sec_sim_6.h5', 'w') as hf:
        hf.create_dataset("<F>",  data=F[:,0])     
        hf.create_dataset("rabi", data = params["rabi"])
        hf.create_dataset("larmor", data = params["larmor"]) 
        hf.create_dataset("quad", data = params["quad"])               
        hf.create_dataset("init state", data = [0,1,0])

    #atom.bloch_plot(pts)

    #atom.proj_plot(proj = 'all')
    '''
    '''
    a = (1e6+1e4)
    b = (1e6-1e4)

    fs = 1e7

    s = F[:,0]


    #s = signal.decimate(s,100)
    t = np.arange(0,0.1, 1e-7)
    if len(s) != len(t):
        s = s[:-1]

    r1c = np.cos(2*np.pi*a*t)
    r1s = np.sin(2*np.pi*a*t)
    r2c = np.cos(2*np.pi*b*t)   
    r2s = np.sin(2*np.pi*b*t)

    #Low frequency signal
    m3 = np.multiply(s,r1c)
    m4 = np.multiply(s,r1s)
    m3 = LPFilter(m3, 1e4, fs)
    m4 = LPFilter(m4, 1e4, fs)

    #High frequency signal
    m5 = np.multiply(s,r2c)
    m6 = np.multiply(s,r2s)
    m5 = LPFilter(m5, 1e4, fs)
    m6 = LPFilter(m6, 1e4, fs)

    np.savetxt('Fx.txt', (s,m6))
    #r49 = np.sin(2*np.pi*a*t)
    #r50 = np.cos(2*np.pi*a*t)
    #r51 = np.cos(2*np.pi*b*t)   
    #r100 = np.sin(2*np.pi*b*t)

    


    
    plt.figure()
    plt.plot(t,m3)
    plt.plot(t,m4)
    plt.plot(t,m5)
    plt.plot(t,m6)
    plt.title('Demodulated Fx')
    plt.xlabel('t (s)')
    plt.ylabel('demodulated <Fx> component')
    plt.legend(['cos component 1.01MHz', 'sin component 1.01MHz','cos component 0.99MHz','sin component 0.99MHz'])
    plt.show()
        
    f = getNormalModes(t, m3)
    print('m3', f[2])
    plt.figure()
    plt.xlim(0,200)
    plt.plot(f[0],f[1])
    plt.title('1.01MHz sin')
    plt.show()
    f = getNormalModes(t, m4)
    print('m4', f[2])
    plt.figure()
    plt.title('1.01MHz cos')
    plt.xlim(0,200)
    plt.plot(f[0],f[1])
    plt.show()
    f = getNormalModes(t, m5)
    print('m5', f[2])
    plt.figure()
    plt.title('0.99MHz sin')
    plt.xlim(0,200)
    plt.plot(f[0],f[1])
    plt.show()
    f = getNormalModes(t, m6)
    print('m6', f[2])
    plt.figure()
    plt.xlim(0,200)
    plt.title('0.99MHz Cos')
    plt.plot(f[0],f[1])
    plt.show()
    '''

    '''
    fs = 1e7
    f, t1, Sxx = signal.spectrogram(F[:,0], fs, nperseg = int(0.1e5),noverlap=int(0.05e5), nfft=int(0.6e5))

    plt.figure()
    plt.pcolormesh(t1,f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim(9e5,11e5)
    plt.show()
    
    '''
    



    #print(t)
    #print(t1)
    #print(Sxx)
    #plt.figure()
    #plt.plot(t, Sxx)

    #plt.show()

    #atom.exp_plot(op = 'F')
    #atom.maj_vid(pts)
    #atom.proj_plot(proj = 'all')
    #atom.bloch_plot(pts)
    #atom.maj_vid(pts)



    #atom.norm_plot()

    '''
    #print(params["rabi"])
    #atom.bloch_plot(pts)
    params["rabi"] = 1e4
    atom.reset()
    t, F, states = atom.state_evolve(params=params, bloch=[False, 101])
    atom.exp_plot(op = 'F')
    atom.proj_plot(proj = 'all')
    '''
    '''
    params["rff"] = 1e6+1e3
    atom.reset()
    t, F, states, pts = atom.state_evolve(params=params, bloch=[True, 100])
    #atom.exp_plot(op = 'F')
    atom.proj_plot(proj = 'all')
    atom.bloch_plot(pts)

    '''
    '''    



    time1, pp, Bfield1 = atom.field_get(params)
    plt.plot(time1, Bfield1[:,2])
    plt.show()
    # exit()
    ''' 

    #print(states)
    #atom.exp_plot(op=  'x',title = "Evolution of <F_x>" , ylim = [-1.05, 1.05])
    #atom.exp_plot(op=  'y',title = "Evolution of <F_y>" , ylim = [-1.05, 1.05] )
    #atom.exp_plot(op=  'z',title = "Evolution of <F_z>" , ylim = [-1.05, 1.05] )
    '''atom.proj_plot(proj = np.array([1,0,0]), ylim = [0, 1.05], title = "|c_a|^2")
    atom.proj_plot(proj = np.array([0,1,0]), ylim = [0, 1.05], title = "|c_b|^2" )
    atom.proj_plot(proj = np.array([0,0,1]), ylim = [0, 1.05], title = "|c_c|^2" )'''
    #atom.proj_plot(proj = np.array([0,0,1]), ylim = [0, 1.05], title = "|c_c|^2" )
    #
    #print(states)
    #states = np.ndarray.tolist(states)

