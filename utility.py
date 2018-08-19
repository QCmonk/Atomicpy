import numpy as np
from numpy.linalg import eig, inv
from operators import *
import scipy.sparse as ssp
import matplotlib as mpl
import scipy.linalg as sla
import matplotlib.pyplot as plt


mpl.rcParams['figure.figsize'] = 16, 9


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


def cexpm_fast(h_cache, time, batch_size=500):
    """
    Currently not working and not going to fix because it is slower than piecewise!
    Computes matrix exponential for h_cache in fast batches as exp(h_cache[t]*t).
    Does NOT include complex factors.
    """
    # ensure that h_cache is a sparse matric
    if not isinstance(h_cache, list):
        raise TypeError(
            "Hamiltonian cache must be a list not {}".format(type(h_cache)))
    # ensure time vector dimension equals that of hamiltonian number
    if len(time) != len(h_cache):
        raise ValueError("Dimension mismatch of time vector and number of Hamiltonians: {} != {}".format(
            len(time), len(h_cache)))

    # compute number of batches required
    batch_num = int(np.ceil(len(h_cache)/batch_size))
    u_cache = np.zeros((2,2,len(h_cache)), dtype=np.complex128)
    # compute matrix exponential in batches
    for b in range(batch_num):
        # extract batch sequence and convert to sparse block diagonal
        batch = ssp.block_diag(h_cache[b*batch_size: (b+1)*batch_size])
        # extract time range and double to fit eigenavlue
        t_batch = np.repeat(time[b*batch_size: (b+1)*batch_size], repeats=2)
        # compute eigenvalues and vectors of batch
        e,V = np.linalg.eigh(batch.todense())
        print(batch)

        V = ssp.csc_matrix(V)
        # compute diagonal exponential 
        D = ssp.diags(np.exp(np.multiply(t_batch, e)))
        # compute matrix exponential
        batch_exp = V.dot(D.dot(ssp.linalg.inv(V))) 
        # extract diagonals (retrieving blocks is super annoying)

        # assign main diagonal to cache (even)
        u_cache[0,0,b*batch_size:(b+1)*batch_size] = batch_exp.diagonal()[::2]
        # assign main diagonal to cache (off)
        u_cache[1,1,b*batch_size:(b+1)*batch_size] = batch_exp.diagonal()[1::2]
        # assign off diagonal
        u_cache[0,1,b*batch_size:(b+1)*batch_size] = batch_exp.diagonal(k=1)[::2]
        u_cache[1,0,b*batch_size:(b+1)*batch_size] = batch_exp.diagonal(k=-1)[::2]

    return u_cache


# a = -1j*np.asarray([[gyro/2, 2],[2, -gyro/2]])
# b = [a]*11000
# time = [0.01]*len(b)
# print(sla.expm(a))
# print(cexpm_fast(b,time)[:,:,0])


def expm_eig(A, t):
    """
    Fast computation of exp(A*t) for 2x2 matrices.
    """
    # compute eigenvalues of A
    e, V = eig(A)
    # compute eigenvector inverse
    Vinv = inv(V)
    # compute exponential of eigenvalues
    E = np.zeros((2, 2), dtype=np.complex128)
    for i in range(2):
        E[i, i] = np.exp(e[i]*t)
    return V @ E @ Vinv


def myexpm(m):
    a, b, c, d = m[0,0], m[0,1], m[1,0], m[1,1]
    delta = np.sqrt((a-d)*(a-d) + 4*b*c)
    C1 = np.exp(0.5*(a+d)) / delta
    C2 = np.sinh(0.5*delta)
    C3 = np.cosh(0.5*delta)
    m00 = C1*(delta*C3+(a-d)*C2)
    m01 = 2*b*C1*C2
    m10 = 2*c*C1*C2
    m11 = C1*(delta*C3+(d-a)*C2)
    return np.array([[m00,m01],[m10,m11]])


from timeit import timeit
from scipy.linalg import expm

def expm_eig(A, t):
    """
    Fast computation of exp(A*t) for 2x2 matrices.
    """
    # compute eigenvalues of A
    e, V = eig(A)
    # compute eigenvector inverse
    Vinv = inv(V)
    # compute exponential of eigenvalues
    E = np.zeros((2, 2), dtype=np.complex128)
    for i in range(2):
        E[i, i] = np.exp(e[i]*t)
    return V @ E @ Vinv


def myexpm(m):
    a, b, c, d = m[0,0], m[0,1], m[1,0], m[1,1]
    delta = np.sqrt((a-d)*(a-d) + 4*b*c)
    C1 = np.exp(0.5*(a+d)) / delta
    C2 = np.sinh(0.5*delta)
    C3 = np.cosh(0.5*delta)
    m00 = C1*(delta*C3+(a-d)*C2)
    m01 = 2*b*C1*C2
    m10 = 2*c*C1*C2
    m11 = C1*(delta*C3+(d-a)*C2)
    return np.array([[m00,m01],[m10,m11]])

#print("scipy.linalg.expm run {:d} times takes {:f} seconds".format(timereps, timeit("expm(data[0])", number=timereps, setup="from __main__ import data,expm")))
#print("expm_eig run {:d} times takes {:f} seconds".format(timereps, timeit("expm_fast(data[0],0.01)", number=timereps,setup="from __main__ import data, expm_eig")))
#print("myexpm run {:d} times takes {:f} seconds".format(timereps, timeit("myexpm(data[0]*0.01)", number=timereps, setup="from __main__ import data,myexpm")))

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

    

def pulse_gen(freq=100, tau=[1.0], amp=1):
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

    for time in tau:
        # generate spike functions, avoiding variable reference issue
        terms.append(lambda t, tau=time: box(t, tau, tau+1/freq)
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


def plot_gen_1(freqs, projs, time, signal):
    """
    Generates publication ready plots using compressive atomic sensing
    """
    plt.style.use('dark_background')
    plt.subplot(2, 1, 1)
    plt.plot(time, signal(time), 'g-')
    # plt.grid(True)
    plt.title("Magnetic field signal", fontsize=18)
    plt.ylim([-1.6e-6, 1.6e-6])
    plt.ylabel("Amplitude (G)", fontsize=14)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
    plt.xlabel("Time (s)", fontsize=14)

    plt.subplot(2, 1, 2)
    #plt.stem(freqs, projs, linewidth=0.6, bottom=np.mean(projs))
    plt.plot(freqs, projs, 'o-', linewidth=0.6, alpha=0.3,)
    plt.ylabel("$|\langle 1 | \psi(t) \\rangle |^2 $", fontsize=16)
    plt.xlabel("Tuning frequency (Hz)", fontsize=14)
    plt.savefig("original.png", dpi=2000)
    # plt.grid(True)
    plt.show()


def plot_gen_2(freqs, projs, comp_f, comp_p, time, recon, measurements):
    """
    Generates publication ready plots using compressive atomic sensing
    """
    plt.style.use('dark_background')
    plt.subplot(2, 1, 1)
    plt.plot(time, recon, 'r-')
    plt.title("Magnetic field signal reconstruction with {} Fourier measurements".format(
        measurements), fontsize=18)
    plt.ylabel("Amplitude (G)", fontsize=14)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
    plt.xlabel("Time (s)", fontsize=14)

    plt.subplot(2, 1, 2)

    plt.plot(freqs, projs, 'o-', alpha=0.3, linewidth=0.6, label="_nolegend_")
    plt.plot(comp_f, comp_p, 'r*', linewidth=1.5, label="Sample frequencies")
    plt.legend()
    plt.ylabel("$|\langle 1 | \psi(t) \\rangle |^2 $", fontsize=16)
    plt.xlabel("Tuning frequency (Hz)", fontsize=14)
    plt.savefig("reconstruction.png", dpi=2000)
    plt.show()
