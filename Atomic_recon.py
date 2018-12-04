from capy import *
from operators import *
from qChain import *
from utility import *
from IPython import embed

import os
import time
import numpy as np 
import scipy.signal as scs
import matplotlib.pyplot as plt

# TODO: figure save and reconstruction save
#       multiprocessing add
#       Add metric information about reconstruction to plots





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
    params = {"tstart":        0,              # time range to simulate over
              "tend":       1e-3,
              "dt":         1e-8,
              "larmor":     gyro,              # bias frequency (Hz)
              "rabi":       1001,              # dressing amplitude (Hz/2)
              "rff":        gyro,              # dressing frequency (Hz)
              "nf":         1000,              # neural signal frequency (Hz)
              "sA":            0,              # neural signal amplitude (Hz/G)
              "nt":         0.00,              # neural signal time event (s)
              "nte":        0.09,
              "dett":        0.1,              # detuning sweep start time
              "detA":       2000,              # detuning amplitude
              "dete":       0.49,              # when to truncate detuning
              "beta":         40,              # detuning temporal scaling
              "xlamp":         0,              # amplitude of 
              "xlfreq":     1000,
              "xlphase":     0.0,
              "zlamp":         0,
              "zlfreq":     1000,
              "zlphase":     0.0,
              "proj": meas1["0"],              # measurement projector
              "savef":       100}              # plot point frequency
 

    shift = (params['rabi']**2)/(4*params["rff"]) + (params["rabi"]**4/(4*(params["rff"])**3))
    params["larmor"] += shift

    atom = SpinSystem(init="zero")
    tdomain, probs = atom.state_evolve(params=params, bloch=[False, 1], lite=False)
    
    #atom.bloch_plot(pnts)

    # time1, pp, Bfield1 = atom.field_get(params)
    # plt.plot(time1, Bfield1[:,1])
    # plt.show()
    # exit()
    print(atom.probs[-1])
    atom.exp_plot(tdomain, probs, "Evolution of <F_z>")



