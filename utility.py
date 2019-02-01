import numpy as np
from scipy.signal import hilbert
from numpy.linalg import eigh, inv
from operators import *
import scipy.sparse as ssp
import matplotlib as mpl
import scipy.linalg as sla
import scipy.signal
from time import gmtime
import json
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = 16, 9


# TODO: Epsilon reconstruction search (100 values)
#       Error metric for reconstructions (RMSE?)



def dict_write(path, rdict):
    """
    Writes a dictionary to a text file at location <path>
    """
    date = gmtime()
    name = "{}_{}_{}_{}_{}_{}".format(date.tm_year,date.tm_mon, date.tm_mday, date.tm_hour, date.tm_min, date.tm_sec)

    # convert everything to a fucking string
    for key in rdict:
        rdict[key] = str(rdict[key])

    # write to file
    with open(path+name+".json", 'a') as file:
        json.dump(rdict, file)
    
    return name

# define rectangular function centered at 0 with width equal to period
def rect(period, time):
    return np.where(np.abs(time) <= period/2, 1, 0)


# hyperbolic secant
def sech(x):
    return 1/np.cosh(x)


def remap(state):
    """
    converts a quantum state to bloch vector
    """
    mstate = np.matrix(state)
    rho = np.outer(mstate.H, mstate)
    u = 2*np.real(rho[0, 1])
    v = 2*np.imag(rho[1, 0])
    w = np.real(rho[0, 0] - rho[1, 1])
    return [u, v, w]


def conjt(A):
    """
    returns conjugate transpose of A
    """
    return np.conj(np.transpose(A))


def normalisedQ(state):
    """
    Checks if an input state is correctly normalised
    """
    # check if vector or operator
    if np.shape(state)[0] == np.shape(state)[1]:
        return np.isclose(np.trace(state), 1.0)
    else:
        # compute sum of probability distribution
        psum = 0.0
        for i in state:
            psum += np.abs(i)**2
        return np.isclose(psum, 1.0)


def unitary_concat(cache, index):
    unitary = cache[0]
    for u in cache[1:index]:
        unitary = u @ unitary
    return unitary



def dict_sanitise(params):
    """
    Function that ensures no Value errors for C dict retrieval
    """
    # set parameter dictionary to defaults if not specified
    cparams = {"tstart": 0.0,
               "tend":  1e-4,
               "dt":    1e-5, 
               "savef": 1,
               "larmor": gyro, 
               "phi": 0, 
               "rabi": 0, 
               "rff": 0, 
               "rph": 0,
               "proj": meas1["0"], 
               "quad": 0,
               "dett": 1e4,
               "detA": 0,
               "dete": 1,
               "xlamp": 0.0,
               "xlfreq": 50,
               "xlphase": 0.0,
               "tr2": 0.0,
               "zlamp": 0.0,
               "zlfreq": 50,
               "zlphase": 0.0,
               "beta": 10, 
               "nt": 1e4, 
               "nf": 1, 
               "sA": 0.0,
               "cdtrabi": 0.0,
               "sdtrabi": 0.0,
               "ctcrabi": 1.0,
               "stcrabi": 0.0,
               "cdtxl": 0.0,
               "sdtxl": 0.0,
               "ctcxl": 1.0,
               "stcxl": 0.0,
               }

    # overwrite defaults
    for p in cparams.keys():
        # default value to zero (kills all not specified components)
        if p not in params.keys():
            cparams[p] = 0.0
        else:
            cparams[p] = params[p]

    # set end spike default so a single spike occurs
    if 'nte' not in cparams.keys():
        cparams["nte"] = cparams["nt"] + 1/cparams["nf"]    

    return cparams


def state2pnt(cache):
    """
    maps a state cache to a set of bloch vector points suitable for plotting
    """
    x, y, z = [], [], []
    pnts = []
    for i in cache[::2]:
        pnts.append(remap(np.matrix(i)))
    for vec in pnts:
        x.append(vec[0])
        y.append(vec[1])
        z.append(vec[2])

    return [x, y, z]


def frame_analyser(state_cache, frame=["interaction", "lab"], time=None, larmor=gyro, omega=1e4, detuning=1e-9, project=meas1["0"]):
    """
    Investigates the state evolution of a system in the lab frame in 
    terms of more interesting ones.
    """
    # determine dimensionality of quantum system
    dim = np.shape(state_cache[0])[0]
    # time step size
    dt = time[1] - time[0]

    # compute reference frame map (bit crude but sufficient for its temporary use)
    if callable(frame):
        assert time is not None, "No time vector supplied for callable unitary transform"
        unitary_map = frame
    elif frame == "interaction":
        def unitary_map(t, larmor=larmor):
            return np.asarray([[np.exp(1j*np.pi*larmor*t), 0], [0, np.exp(-1j*np.pi*larmor*t)]])
    elif frame == "dressed":
        # define dressed state transform
        def dressed(t, omega=omega, detuning=detuning): return np.asarray([[np.cos(np.arctan(
            omega/detuning)/2), -np.sin(np.arctan(omega/detuning)/2)], [np.sin(np.arctan(omega/detuning)/2), np.cos(np.arctan(omega/detuning)/2)]])

        def unitary_map(t, larmor=larmor): return dressed(t) @ np.asarray([[np.exp(1j*np.pi*larmor*t), 0], [0, np.exp(-1j*np.pi*larmor*t)]])
    else:
        raise ValueError("Unrecognised reference frame")

    # apply transform to states
    if callable(unitary_map):
        new_states = [unitary_map(t) @ state_cache[:, :, step] for step, t in enumerate(time)]
        # for step,t in enumerate(time):
        #    print(unitary_map(step) @ np.asarray([[1],[1]])/np.sqrt(2))
    else:
        new_states = [unitary_map @ state for state in state_cache]

    # compute projectors
    nprobs = np.squeeze([np.abs(project @ nstate)**2 for i, nstate in enumerate(new_states)])
    if time is not None:
        # plot new probabilities
        plt.plot(time, nprobs)
        plt.ylim([0, 1])
        plt.grid()
        plt.show()

    return state2pnt(new_states)


def stability_measure(time, omega, detune):
    """
    Generates a plot of the derivative of the precession angle of a state
    in the presence of a detuning field and compares it against the Rabi vector
    as a function of time
    """
    pass

    
def demodulate(time, signal, cfreq, fs):
    """
    Performs simple demodulation of an input signal given carrier
    frequency <cfreq> and sample rate <fs>.
    """
    
    carrier_cos = np.cos(2*np.pi*cfreq*time)
    carrier_sin = np.sin(2*np.pi*cfreq*time)

    csignal = carrier_cos*signal
    ssignal = carrier_sin*signal

    # simple butterworth with cutoff at 2*cfreq
    Wn = 2*cfreq/fs
    b,a = scipy.signal.butter(3, Wn)


    filt_cos = scipy.signal.lfilter(b,a, csignal)
    filt_sin = scipy.signal.lfilter(b,a, ssignal)


    plt.plot(time, filt_cos, label='Filtered Cosine')
    plt.plot(time, filt_sin, label='Filtered Sine')
    plt.grid(True)
    plt.legend()
    plt.show()



def pulse_gen(freq=100, tau=[1.0], amp=1, nte=[None]):
    """
    Generates a multipulse signal with a pretty neat dynamic approach
    """

    # define a box function
    def box(t, start, end):
        if not isinstance(t, np.ndarray):
            t = np.asarray(t)

        return (t > start) & (t < end)

    # define sum function
    def sig_sum(t, terms):
        return sum(f(t) for f in terms)

    # list to hold spike functions
    terms = []

    for time,end in zip(tau,nte):
        if end is None:
            # generate spike functions, avoiding variable reference issue
            terms.append(lambda t, tau=time: box(t, tau, tau+1/freq)
                         * amp*np.sin(2*np.pi*freq*(t-tau)))
        else:
            # generate spike functions, avoiding variable reference issue
            terms.append(lambda t, tau=time: box(t, tau, end)
                         * amp*np.sin(2*np.pi*freq*(t-tau)))

    # generate final signal function
    signal = lambda t, funcs=terms: sig_sum(t, funcs)

    return signal


def input_output(f_range=[200, 400, 2], sig_range=[300, 400, 2], t=[0, 1/10, 1/1e3]):
    """
    Computes a set of input/output transformation pairs as part of the map determination problem
    """
    # number of signal samples to generate
    K = len(np.arange(sig_range[0], sig_range[1], sig_range[2]))
    # number of measurements to make
    M = len(np.arange(f_range[0], f_range[1], f_range[2]))
    # length of signal vector
    N = len(np.linspace(t[2], t[1], int((t[1] - t[0])/t[2])))
    # sensor array
    sensor = np.zeros((M, K), dtype=float)
    # source array
    source = np.zeros((N, K), dtype=float)
    # ensure reconstruction is possible
    assert M <= K, "Need more signal data to reconstruct data array"
    assert N == M, "Current investigation method requires same dimensionality"

    # iterate over signal freqs
    for k, sig_freq in enumerate(np.arange(sig_range[0], sig_range[1], sig_range[2])):
        print("Computing sensor output for test signal {}/{}".format(k, K), end='\r')
        # perform map operation
        freqs, projs, signal = pseudo_fourier(
            sig_amp=5, sig_freq=sig_freq, f_range=f_range, plot=False)
        # store output projections and source signal
        sensor[:, k] = projs
        source[:, k] = signal

    return source, sensor


def plot_gen_1(freqs, projs, time, sim_vars):
    """
    Generates publication ready plots using compressive atomic sensing
    """
    signal = pulse_gen(sim_vars["sig_freq"], tau=[sim_vars["tau"]], amp=sim_vars["sig_amp"], nte=[sim_vars["nte"]])

    plt.style.use('dark_background')
    plt.subplot(2, 1, 1)
    plt.plot(time, signal(time), 'g-')
    # plt.grid(True)
    plt.title("Magnetic field signal", fontsize=18)
    plt.ylim([-1, 1])
    plt.ylabel("Amplitude (G)", fontsize=14)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
    plt.xlabel("Time (s)", fontsize=14)

    plt.subplot(2, 1, 2)
    #plt.stem(freqs, projs, linewidth=0.6, bottom=np.mean(projs))
    plt.plot(freqs, projs, 'o-', linewidth=0.6, alpha=0.3,)
    plt.ylabel("$|\langle 1 | \psi(t) \\rangle |^2 $", fontsize=16)
    plt.xlabel("Tuning frequency (Hz)", fontsize=14)
    plt.savefig("original.png", dpi=1500)
    # plt.grid(True)
    plt.show()

def signal_generate(time, sim_vars):
    """
    Generates the neural signal defined by sim_vars
    """
    # assume single pulse if not defined
    if "nte" not in sim_vars.keys():
        nte = sim_vars["tau"] + 1/sim_vars["sig_freq"]
        sim_vars["nte"] = nte

    signal = pulse_gen(sim_vars["sig_freq"], tau=[sim_vars["tau"]], amp=sim_vars["sig_amp"], nte=[sim_vars["nte"]])
    signal = signal(time)
    signal /= np.max(np.abs(signal))

    return signal


def rmse(v1,v2):
    """
    Computes RMSE between two input vectors
    """
    return np.sqrt(np.mean((v1-v2)**2))

def plot_gen_2(freqs, projs, comp_f, comp_p, time, recon, sim_vars, savefig=False):
    """
    Generates publication ready plots using compressive atomic sensing
    """
    # generate original signal
    signal = pulse_gen(sim_vars["sig_freq"], tau=[sim_vars["tau"]], amp=sim_vars["sig_amp"], nte=[sim_vars["nte"]])
    signal = signal(time)
    signal /= np.max(np.abs(signal))
    recon /= np.max(np.abs(recon))

    # format projections to expectation value
    projs = 2*projs - 1
    comp_p = 2*comp_p - 1
    # get number of measuremens used
    measurements = sim_vars["measurements"]

    #plt.style.use('dark_background')
    plt.subplot(2, 1, 1)
    plt.plot(time, signal, 'g-')
    plt.plot(time, recon, 'r-')
    plt.legend(["Original","Reconstruction"])
    plt.title("Magnetic field signal reconstruction with {} Fourier measurements using \"{}\" method".format(
        measurements, sim_vars["method"]), fontsize=18)
    plt.ylabel(r"Amplitude", fontsize=14)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
    plt.xlabel("Time (s)", fontsize=14)

    plt.subplot(2, 1, 2)

    plt.plot(freqs, projs, 'o-', alpha=0.3, linewidth=0.6, label="_nolegend_")
    plt.plot(comp_f, comp_p, 'r*', linewidth=1.5, label="Sample frequencies")
    plt.legend()
    plt.ylabel(r"$\langle F_z \rangle  $", fontsize=16)
    plt.xlabel("Tuning frequency (Hz)", fontsize=14)
    plt.figure(num=1, figsize=[12,9])
    if savefig:
        path = "C:\\Users\\Joshua\\Research\\Uni\\2018\\Project\\Latex\\Proj_Figures\\"
        # save parameters to text file
        fig_name = dict_write(path, sim_vars)
        plt.savefig(path+fig_name+".png", transparent=True, dpi=1000)

    plt.show()