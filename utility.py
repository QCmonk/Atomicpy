import numpy as np 
import matplotlib.pyplot as plt


# define rectangular function centered at 0 with width equal to period
def rect(period, time):
    return np.where(np.abs(time) <= period/2, 1, 0)


# hyperbolic secant
def sech(x):
    return 1/np.cosh(x)


def pulse_gen(freq=100, tau=[1.0], amp=1):
    """
    Generates a multipulse signal with a pretty neat dynamic approach
    """

    # define a box function
    def box(t, start, end):
        if not isinstance(t, np.ndarray):
            t = np.asarray(t)

        return (t>start) & (t<end)

    # define sum function
    def sig_sum(t, terms):
        return sum(f(t) for f in terms)

    # list to hold spike functions
    terms = []

    for time in tau:
        # generate spike functions, avoiding variable reference issue
        terms.append(lambda t, tau=time: box(t, tau, tau+1/freq)*amp*np.sin(2*np.pi*freq*(t-tau)))

    # generate final signal function
    signal = lambda t, funcs=terms: sig_sum(t, funcs)

    return signal



def input_output(f_range=[200,400,2], sig_range=[300, 400, 2], t=[0, 1/10, 1/1e3]):
    """
    Computes a set of input/output transformation pairs as part of the map determination problem
    """
    # number of signal samples to generate
    K = len(np.arange(sig_range[0], sig_range[1], sig_range[2]))
    # number of measurements to make
    M = len(np.arange(f_range[0],f_range[1], f_range[2]))
    # length of signal vector
    N = len(np.linspace(t[2], t[1], int((t[1] - t[0])/t[2])) )
    # sensor array
    sensor = np.zeros((M,K), dtype=float)
    # source array
    source = np.zeros((N,K), dtype=float)
    # ensure reconstruction is possible
    assert M <= K, "Need more signal data to reconstruct data array" 
    assert N == M, "Current investigation method requires same dimensionality"

    # iterate over signal freqs
    for k,sig_freq in enumerate(np.arange(sig_range[0], sig_range[1], sig_range[2])):
        print("Computing sensor output for test signal {}/{}".format(k,K), end='\r')
        # perform map operation
        freqs, projs, signal = pseudo_fourier(sig_amp=5, sig_freq=sig_freq, f_range=f_range, plot=False)
        # store output projections and source signal
        sensor[:,k] = projs
        source[:,k] = signal

    return source, sensor






def plot_gen_1(freqs, projs, time, signal):
    """
    Generates publication ready plots using compressive atomic sensing
    """
    plt.style.use('dark_background')
    plt.subplot(2,1,1)
    plt.plot(time, signal(time), 'g-')
    #plt.grid(True)
    plt.title("Magnetic field signal", fontsize=18)
    plt.ylim([-1.6e-6, 1.6e-6]) 
    plt.ylabel("Amplitude (G)", fontsize=14)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,3))
    plt.xlabel("Time (s)",fontsize=14)


    plt.subplot(2,1,2)
    #plt.stem(freqs, projs, linewidth=0.6, bottom=np.mean(projs))
    plt.plot(freqs, projs, 'o-', linewidth=0.6)
    plt.ylabel("$|\langle 1 | \psi(t) \\rangle |^2 $", fontsize=16)
    plt.xlabel("Tuning frequency (Hz)", fontsize=14)
    #plt.grid(True)
    plt.show()



def plot_gen_2(freqs, projs, comp_f, comp_p, time, recon, measurements):
    """
    Generates publication ready plots using compressive atomic sensing
    """
    plt.style.use('dark_background')
    plt.subplot(2,1,1)
    plt.plot(time, recon, 'r-')
    plt.title("Magnetic field signal reconstruction with {} Fourier measurements".format(measurements), fontsize=18)
    plt.ylabel("Amplitude (G)", fontsize=14) 
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,3))
    plt.xlabel("Time (s)", fontsize=14)

    plt.subplot(2,1,2)

    plt.plot(freqs, projs, 'o-', alpha=0.3, linewidth=0.6, label="_nolegend_")
    plt.plot(comp_f, comp_p, 'rp', linewidth=1.5, label="Sample frequencies")
    plt.legend()
    plt.ylabel("$|\langle 1 | \psi(t) \\rangle |^2 $", fontsize=16)
    plt.xlabel("Tuning frequency (Hz)", fontsize=14)
    plt.show()
