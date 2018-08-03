from capy import *
from qChain import *
from utility import *
from operators import *
import numpy as np 
import matplotlib.pyplot as plt

# TODO:
# Fix up the parameter logic - it is pretty fucked up right now


def recon_pulse(sim_vars):
    # compute full measurement record in the frequency range f_range - gives Fourier Sine coefficients
    bfreqs, projs, signal = pseudo_fourier(struct="pulse", 
                                          sig_amp=sim_vars["sig_amp"], 
                                          sig_freq=sim_vars["sig_freq"], 
                                          f_range=sim_vars["frange"], 
                                          tau=sim_vars["tau"], 
                                          noise=sim_vars["noise"], 
                                          t=sim_vars["time"],
                                          plot=False, 
                                          verbose=True)        


    # add to simulation parameter dictionary if not already defined
    if "measurements" not in sim_vars.keys():
      sim_vars["measurements"] = 100

    # determine time vector and add to variable dictionary 
    time = np.arange(0, sim_vars["time"], 1/sim_vars["fs"])
    sim_vars["time"] = time

    # generate measurement transform
    transform, ifreqs, _ = measure_gen(ndim=len(time), 
                                    time=time, 
                                    basis="fourier", 
                                    measurements=sim_vars["measurements"], 
                                    freqs=bfreqs)
    
    # compute measurement record, selecting only those chosen for reconstruction
    meas_record = projs[[i for i in ifreqs]] - 0.5 # (projs[comp.rand_ints]-0.5)/(np.max(projs[comp.rand_ints] -0.5))

    #plt.plot((transform @ signal(time)/np.max(transform @ signal(time))))
    #plt.plot(meas_record/np.max(meas_record))
    #plt.show()



    # create optimiser instance
    comp = CAOptimise(svector=meas_record,   # measurement vector in sensor domain
                      transform=transform,   # sensor map from source to sensor
                      verbose=True,          # give me all the information
                      **sim_vars)           # epsilon and other optional stuff

    comp.cvx_recon()

    # measurement frequencies used
    comp_f = bfreqs[ifreqs]
    # measurement record adjustment
    comp_p = meas_record + 0.5
    # signal reconstruction
    recon = comp.u_recon

    plot_gen_1(bfreqs, projs, time, signal)
    plot_gen_2(bfreqs, projs, comp_f, comp_p, time, recon, measurements=sim_vars["measurements"])




if __name__ == "__main__":


    ################################################################
    # Compressive Reconstruction stuff
    ################################################################
    # np.random.seed(141) #  seed 131 fails with 50 measurements

    # # define user parameters for simulation run
    # sim_vars = {"measurements":      20,          # number of measurements
    #             "epsilon":         0.01,          # radius of hypersphere in measurement domain
    #             "basis":      "fourier",          # measurement basis to use
    #             "sig_amp":            1,          # amplitude of magnetic field signal in Gauss/Hz
    #             "sig_freq":         350,          # frequency of magnetic field signal
    #             "tau":           [5e-2],          # time events of pulses
    #             "frange":   [50,500,2],          # frequency tunings of BECs
    #             "noise":           0.00,          # noise to add to measurement record SNR percentage e.g 0.1: SNR = 10
    #             "time":             0.2,          # time to simulate atomic evolution for
    #             "fs":               2e3}          # sampling rate for measurement transform

    # recon_pulse(sim_vars)
    ################################################################


    # set detune max value
    detune_max = 1e4
    # time at which detuning sweep occurs
    detune_tau = 1e-3
    # time for sweep to occur
    swt = 0.05
    # bias field strength
    bias_amp = gyro
    # coupling field strength
    #omega1 = lambda t: np.clip((1e4 - np.heaviside(detune_tau, 1.0)*(t-detune_tau)*detune_max/swt), 0.0, None)
    omega1 = lambda t: np.clip((1e4*np.heaviside(-(t-detune_tau), 0.0) + 1e4*np.heaviside(t-detune_tau, 1.0)*sech(3*np.pi*(t-detune_tau)/swt))/gyro  , 0.0, None)

    # define detuning function 
    #delta = lambda t: (np.heaviside(t-detune_tau, 1.0)*(t-detune_tau)*detune_max/swt)/gyro
    delta = lambda t: (np.heaviside(t-detune_tau, 1.0)*detune_max*np.tanh(3*np.pi*(t-detune_tau)/swt)/gyro)

    # create annoyingly complex Fx magnetic field function
    wrf = lambda t: bias_amp/gyro

    # create Fx field
    Fx = lambda t: 1e4*np.cos(2*np.pi*wrf(t)*t)/gyro
    Fz = lambda t: bias_amp/gyro #- delta(t)

    # time = np.linspace(0,0.06, 1e4)
    # rabi = np.sqrt((delta(time))**2 + (omega1(time))**2)

    # plt.plot(time, rabi)
    # plt.show()
    # exit()

    # define generic Hamiltonian parameters with Zeeman splitting and rf dressing
    params = {"struct": ["sinusoid",                              # Fx is a sinusoidal dressing field field
                         "constant",                              # Fy is a constant field
                         "constant"],                                # Fz is a fade field 
                         "freqb": [gyro/gyro, 0, 0],                   # frequency in Hz of each field vector
                         "tau": [None, None, 1e-4],               # time event of pulse along each axis
                         "amp":[1e4/gyro, 0/gyro, gyro/gyro],        # amplitude in Gauss -> 1 Gauss ~= 700000 Hz precession
                         "misc": [None, None, None]}          # misc parameters

    # create specified magnetic fields and resulting Hamiltonian
    fields = field_gen(field_params=params)
    ham = Hamiltonian()
    ham.generate_field_hamiltonian(fields)

    #field_plot(fields, time=np.linspace(0,1e-2,1e5))
    #exit()

    # initialise spin system in up state
    atom = SpinSystem(init="zero")

    # evolve state under system Hamiltonian
    time, probs, pnts = atom.state_evolve(t=[1e-6, 1e-4, 5e-7],             # time range and step size to evolve for
                                          hamiltonian=ham.hamiltonian,   # system Hamiltonian
                                          project=meas1["0"],            # projection operator for measurement
                                          bloch=[True, 20])               # Whether to save pnts for bloch state 
                                                                         # and save interval

    atom.bloch_plot(pnts)                                                                    
    atom.prob_plot(time, probs)                  

                                        


  


