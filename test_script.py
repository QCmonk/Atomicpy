from capy import *
from operators import *
from qChain import *
from utility import *
import numpy as np 
import time
import matplotlib.pyplot as plt



if __name__ == "__main__":

    # sampling frequncy of unitary (Hz)
    fs = 1e8
    # time to evolve state for (seconds) 
    eperiod = 1e-4

    # define B fields 
    def Bx(t): return 2*1e4*np.cos(2*np.pi*gyro*t)
    def By(t): return 0*(t/t)
    def Bz(t): return gyro*(t/t)


    # define generic Hamiltonian parameters with Zeeman splitting and rf dressing
    params = {"struct": ["custom",           # Fx is a sinusoidal dressing field field
                         "custom",           # Fy is a constant field
                         "custom"],          # Fz is a fade field 
              "freqb":  [0, 0, 0],           # frequency in Hz of each field vector
              "tau":    [None, None, None],  # time event of pulse along each axis
              "amp":    [0, 0, 0],           # amplitude in Gauss -> 1 Gauss ~= 700000 Hz precession
              "misc":   [Bx, By, Bz]}        # misc parameters

    # create specified magnetic fields and resulting Hamiltonian
    fields = field_gen(field_params=params)
    ham = Hamiltonian() 
    ham.generate_field_hamiltonian(fields)

    # initialise spin 1/2 system in zero (down) state
    atom = SpinSystem(init="zero")

    # evolve state under system Hamiltonian
    start = time.time()
    tdomain, probs, _ = atom.state_evolve(t=[1e-44, eperiod, 1/fs],          # time range and step size to evolve for
                                          hamiltonian=ham.hamiltonian_cache, # system Hamiltonian
                                          cache=True,                        # whether to cache calculations (faster)
                                          project=meas1["0"],                # projection operator for measurement
                                          bloch=[False, 100])                # Whether to save pnts for bloch state 
                                                                             # and save interval
    end = time.time()

    print("Atomic state evolution over {} seconds with Fs = {:.2f} MHz took {:.3f} seconds".format(eperiod, fs/1e6, end-start))
    
    # plot |<0|psi(t)>|^2 against time
    atom.prob_plot(tdomain, probs)