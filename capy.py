import os
import sys
import random
import numpy as np
import cvxpy as cvx
import scipy.sparse as sp
from decimal import Decimal
import matplotlib.pyplot as plt

# TODO
# adaptive template update preprocess
# threshold limit for spike matching
# update measure_gen with fourier measurement



class CAOptimise(object):
    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """

    def __init__(self, svector, transform, verbose=False, **kwargs):
        # the measured vector obtained from experiment
        self.svector = np.asarray(svector, dtype=np.float).reshape([1,-1])
        # dimensionality of measurement basis
        self.mdim = len(self.svector)
        # dimensionality of signal basis
        self.ndim = len(transform.T)
        # check for verbosity level
        self.verbose = verbose
        # sensing flags for debug purposes
        self.flags = []

        # extract keyword arguments after setting defaults
        self.opt_params = {"template": None,
                           "epsilon": 0.01, 
                           "transform": transform,
                           "length": len(self.svector)}
        for key, value in kwargs.items():
            self.opt_params[key] = value


        # check for supplied measurement transform
        if "transform" not in self.opt_params.keys():
            raise KeyError("No transform specified, aborting")
        else:
            self.transform = self.opt_params["transform"]


    def cvx_recon(self):
        """
        Sets up optimisation problem and computes x such that: transform*x - svector = 0.

        Parameters
        ----------
        None

        Returns
        -------
        u_recon : one dimensional numpy array 
            Vector that optimises the compressive sensing problem. 
        """

        # set cvx start flag
        self.flags.append("cvx_recon_start")

        # setup SDP for Ax-b=0 using cvxpy 
        if self.verbose: print("Setting up problem")
        A = self.transform 
        b = self.svector
        x = cvx.Variable(len(self.transform.T))
        objective = cvx.Minimize(cvx.norm(x, 1))
        constraints = [cvx.norm(A*x - b.T, 2) <= self.opt_params["epsilon"]]
        prob = cvx.Problem(objective, constraints)

        # solve 2nd order cone problem
        if self.verbose: print("Solving using CVXOPT")
        prob.solve()

        # print solution status
        if self.verbose:   
            print("Solution status:", prob.status)
            print("Objective: ", prob.value)


        # set cvx end flag
        self.flags.append("cvx_recon_end")
        # compute reconstruction
        self.u_recon = x.value/np.max(x.value)
        # store error vector
        self.u_error = self.transform @ self.u_recon - self.svector
        # compute error using both l1 and l2 norm
        self.metrics = {'l1': np.linalg.norm(self.u_error, ord=1), "l2": np.linalg.norm(self.u_error, ord=2)}

        return self.u_recon

    # find time index of most prominent spike using python optimisation
    def py_match(self, osignal=None, plot=False):
        """
        Performs single pass matched filtering using a measurement basis transform

        Parameters
        ----------
        osignal : one dimensional numpy array
            The original signal that is being reconstruced, used for fidelity testing
        
        plot : Boolean  
            Whether to plot the autocorrelation function and osignal if 
            provided.

        Returns
        -------
        The correlation function h(tau).

        Raises
        ------
        AttributeError
            if no template for matching has been provided
        """

        # ensure a template has been provided
        if self.opt_params["template"] is None:
            raise(AttributeError, "No template provided for matched filter")

        # set single shot matched filter flag
        self.flags.append("comp_match_single_start")
        # set time range
        tau_range = self.opt_params["time"]

        # status message
        if self.verbose: print("Performing compressive matched filtering of single spike")

        # define autocorrelation function
        tau_match = lambda tau_int: np.abs(self.svector @ self.transform @ self.sig_shift(self.opt_params["template"], tau_int))
        
        # preallocate correlation vector
        self.correlation = np.zeros((self.ndim,))
        for step, tau in enumerate(tau_range):
            # compute correlation given some time shift of template vector
            self.correlation[step] = tau_match(step)

        # set single shot matched filter flag
        self.flags.append("comp_match_single_end")

        # plot correlation if requested against original signal if supplied
        if plot:
            if osignal is not None:
                fig, axx = plt.subplots(2, sharex=True)
                axx[1].plot(tau_range, self.correlation, 'r')
                axx[0].grid(True)
                axx[0].set_title("Correlation using template")
                axx[1].set_xlabel("Time (s)")
                axx[0].plot(tau_range, osignal)
                axx[1].grid(True)
                plt.show()
            else:
                plt.plot(tau_range, self.correlation, 'r')
                plt.title("Correlation using template")
                plt.xlabel("Time (s)")
                plt.ylabel("Correlation")
                plt.grid(True)
                plt.show()

        return self.correlation

    def py_notch_match(self, osignal=None, max_spikes=1, plot=False):
        """
        Applies the notched matched filter to signal identification problem. 

        Parameters
        ----------
        osignal: one dimensional numpy array
            The original signal, only required for comparison plotting
        
        max_spikes : int
            The number of spikes to reconstruct (soon to be changed to threshold)
        
        plot : boolean
            Whether to plot the matched filter results. If osignal is supplied
            this will compare them, else it will just plot the reconstruction. 

        Returns
        -------
        notch: one dimensional numpy array
            a vector with the template placed at the identified time events

        Raises
        ------
        AttributeError
            If no template has been provided. 
        
        """
        
        # ensure a template has been provided
        if self.opt_params["template"] is None:
            raise(AttributeError, "No template provided for matched filter")

        # set multi match start
        self.flags.append('comp_match_multi_start')

        if self.verbose: print("Performing compressive matched filtering")

        # define notch function
        self.notch = np.ones((self.ndim,1), dtype=float)
        # time period to shift over
        tau_range = self.opt_params["time"]
        # define autocorrelation function
        tau_match = lambda tau_int: np.abs(self.svector @ self.transform @ np.multiply(self.notch ,self.sig_shift(self.opt_params["template"], tau_int)))
        
        spike = 0
        self.template_recon = np.zeros((self.ndim, 1))
        while spike < max_spikes:
            # iterate the number of spikes
            spike += 1

            # compute correlation function over parameter space
            self.correlation = np.zeros((self.ndim,), dtype=float)
            for step, tau in enumerate(tau_range):
                # test = self.transform @ np.multiply(self.notch, self.sig_shift(self.opt_params["template"], step))
                self.correlation[step] = tau_match(step)

            #plt.plot(self.correlation)
            #plt.show()

            # extract maximum peak
            max_ind = np.argmax(a=self.correlation)
            # add peak to vector nuke
            self.notch[max_ind-len(self.opt_params["template"]):max_ind+2*len(self.opt_params["template"])] = 0
            # subtract template from measurement vector
            self.template_recon += self.sig_shift(self.opt_params["template"], max_ind).reshape([-1, 1])

        self.flags.append('comp_match_multi_end')

        # compute error
        self.m_error = self.svector-self.transform @ self.template_recon
        # compute errors
        self.metrics = {'l1': np.linalg.norm(self.m_error, ord=1), "l2": np.linalg.norm(self.m_error, ord=2)}

        # plot reconstruction if requested
        if plot:
            if osignal is not None:
                fig, axx = plt.subplots(2, sharex=True)
                axx[1].plot(tau_range, self.notch, 'r')
                axx[0].grid(True)
                axx[0].set_title("Multievent reconstruction")
                axx[1].set_xlabel("Time (s)")
                axx[0].plot(tau_range, osignal)
                axx[1].grid(True)
                plt.show()
            else:
                plt.plot(tau_range, self.template_recon, 'r')
                plt.title("Correlation using template")
                plt.xlabel("Time (s)")
                plt.ylabel("Correlation")
                plt.grid(True)
                plt.show()

        return self.notch


    def sig_shift(self, template, tau_int):
        """
        Shifts a template by tau_int indexes to the right. 

        Parameters
        ----------
        template : one dimensional numpy array
            The template to use for reconstruction.

        tau_int : int
            The index to insert the template at. 

        Returns
        -------
        shift: one dimensional numpy array
            An all zero vector with template at tau_int.

        """

        # create zeroed vector
        shift = np.zeros((self.ndim,1))
        temp_len = len(template)
        # force clipping
        if tau_int < 0:
            int_pos = 0
        elif tau_int > self.ndim - temp_len:
            tau_int = int(self.ndim - temp_len)
        # shift vector by specified amount
        shift[tau_int: tau_int + temp_len] = template.reshape([-1,1]) 

        return shift


    def plot_recon(self, original):
        """
        Plot function for comparing reconstructed signal and original. 

        Parameters
        ----------
        original : one dimensional numpy array
            The original signal that is being reconstructed

        Returns
        -------
        A plot window showing original signal and reconstruction super imposed on 
        one another. 

        """
        plt.style.use('dark_background')
        plt.plot(self.opt_params["time"], original, color="r", linewidth=1 ,label='Original')
        plt.plot(self.opt_params["time"], self.u_recon, 'b--', linewidth=1, label='Reconstruction')
        plt.title("Compressive sampling reconstruction ")
        #plt.grid(True)
        plt.ylabel("Amplitude")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.savefig("plot_recon.png", dpi=2000)
        plt.show()




    def notch_match_plot(self, time, original):
        """
        Plots time events of signals using (only) notched filter approach and compares against original 
        and notch function.

        Parameters
        ----------
        time : one dimensional numpy array
            time vector (assumed to be time, doesn't have to be) of transform sample time

        original : one dimensional numpy array
            Original signal that is being reconstructed. 

        Returns
        -------
        Plot of notch function, reconstruction and 


        """

        fig, axx = plt.subplots(2, sharex=True)
        l1, = axx[0].plot(time, self.template_recon, 'g')
        axx[0].grid(True)
        axx[0].set_title("Notched matched filter - Samples: {}, Measurements: {}, Basis: {}, Noise Amplitude: {}, :L_2 Error: {}".format(self.opt_params["length"],
                                                                                       self.opt_params["measurements"],
                                                                                       self.opt_params["basis"],
                                                                                       self.opt_params["noise"], 
                                                                                       self.metrics["l2"]))                                     
        l2, = axx[0].plot(time, original, 'r')
        #plt.legend([l1,l2], ["Reconstructed", "Original"])
        l3, = axx[1].plot(time, self.nuke, 'b')
        axx[1].grid(True)
        axx[1].set_xlabel("Time (s)")
        plt.show()


def sparse_gen(events, freq, fs=4e3, t=10, plot=False):
    """
    Generates an example sparse signal for testing and demonstration purposes.

    Parameters
    ----------
    events : int
        Number of events that should be randomly placed over signal period

    freq : float
        The frequency of the signal pulse that occurs at each event (identical for each)

    fs : float
        The sampling frequency of the original signal vector and sampling transform. Only relevant
        for simulations such as this one - must be sufficiently high to capture event information 
        however. 
    
    t : float
        Total time of signal - longer times require more memory (or less measurements)

    plot : boolean
        Whether to plot the signal vector 

    Returns
    -------
    time : one dimensional numpy array
        The time vector used for signal and transform.

    signal : one dimensional numpy array
        The generated sparse signal.

    template : one dimensional numpy array
        The signal template used for each event.

    """

    # define rectangular function centered at 0 with width equal to period
    def rect(period, time):
        return np.where(np.abs(time) <= period/2, 1, 0)

    # initialise time vector
    time = np.arange(0,t,1/fs)

    # generate signal with a single sinusoid of frequency freq
    signal = np.zeros((len(time)))
    for evnt in np.random.choice(time, size=events):
        signal += np.multiply(rect(1/freq, time-evnt),-np.sin(2*np.pi*freq*(time-evnt)))

    # generate template signal for match
    t = np.arange(0, 1/freq, 1/fs)
    template = np.sin(2*np.pi*freq*t)

    # plot the source signal if requested
    if plot:
        plt.plot(time, signal, 'r--')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Initial sparse signal")
        plt.grid(True)
        plt.show()

    return time, signal, template


def measure_gen(ndim, basis="random", time=None, measurements=100, freqs=[100,1000]):
    """
    Generates desired measurement basis set with given parameters.

    Parameters
    ----------
    ndim : int
        The dimension of the original signal vector (number of time samples).

    time : one dimensional numpy array
        The time vector for the original signal vector.

    basis : str
        The measurement basis - random/fourier - to use.
    
    measurements : int
        Number of measurements to use for compressive sampling transform.

    freqs : list (float)
        The range of frequencies to sample when using fourier basis. 

    Returns
    -------
    transform : numpy array
        An measurements*ndim array with basis measurements sorted row wise such that transform*signal = svector  

    """

    if basis == "random":
        transform = 2*np.random.ranf(size=(measurements, ndim))-1

    elif basis == "fourier":
        # pre-allocate transform matrix
        transform = np.zeros((measurements, ndim), dtype=float)

        # generate random frequencies or use those in freq list
        rand_flag = len(freqs) < measurements
        # create storage container for selected indices
        if not rand_flag:
            rand_ints = []

        meas_freq = []
        for i in range(measurements):
            if rand_flag:
                # choose a random frequency over the given range
                freq = (freqs[1] - freqs[0])*np.random.ranf()+freqs[0]
            else:
                # choose a random frequency in the provided set and save index of chosen int
                randint = np.random.randint(low=0, high=len(freqs))
                # ensure frequency has not been chosen before (inefficient but in the scheme of things, unimportant)
                while randint in rand_ints:
                    randint = np.random.randint(low=0, high=len(freqs))
                # save chosen frequency
                rand_ints.append(randint)
                # add to set
                freq = freqs[randint]
            
            # add frequency to selection
            meas_freq.append(freq)
            transform[i, :] = -np.sin(2*np.pi*freq*time) #np.imag(np.exp(-1j*2*np.pi*freq*self.t))

    else:
        print("unknown measurement basis specified: exiting ")
        os._exit(1)
    return transform, rand_ints, meas_freq



