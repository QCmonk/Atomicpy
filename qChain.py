import os
import time
import copy
import quantumc
import numpy as np
from capy import *
from utility import *
from operators import *
from qutip import Bloch
from time import gmtime
from IPython import embed
from hashlib import blake2b
import scipy.linalg as sla
import matplotlib.pyplot as plt

# import without warnings
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

# path to experiment data file
if os.path.isdir('C:\\Users\\Joshua'):
    if os.path.isdir('C:\\Users\\Joshua\\Research'):
        archivepath = 'C:\\Users\\Joshua\\Research\\Code\\Python\\Modules\\qChain\\archive.hdf5'
    else:
        archivepath = 'C:\\Users\\Joshua\\Documents\\Projects\\Uni\\2017\\Research\\markovpy\\archive.hdf5'
else:
    archivepath = 'C:\\Users\\joshm\\Documents\\Projects\\Uni\\2017\\Research\\markovpy\\archive.hdf5'


class SpinSystem(object):
    """
    class that defines a spin-[1/2, 1] system, keeping track of its state and evolution
    """

    def __init__(self, spin: str="half", init=None):
        # spin of system to be defined
        self.spin = spin
        # initial state of each particle [zero, one, super]
        self.init = init

        if self.spin == "half":
            # dimension of system
            self.dim = 2
            # initial state
            self.state = op1["pz"]
        elif self.spin == "one":
            # dimension of system
            self.dim = 3
            # initial state
            self.state = op2["po"]

        if self.init is not None:
            if type(self.init) is str:
                if self.init is "super":
                    self.state = op1["h"] @ self.state
                elif self.init is "isuper":
                    self.state = op1["s"] @ op1["h"] @ self.state
                elif self.init is "zero":
                    self.state = op1["pz"]
                elif self.init is "one":
                    self.state = op1["po"]
                else:
                    raise ValueError(
                        "Unrecognised initial state: {}".format(self.init))
            else:
                # custom start state
                self.state = np.matrix(self.init)

        self.state = np.copy(self.state)

        # keep local copy of initial state
        self.init_state = np.copy(self.state)

    def reset(self):
        """
        Reset atomic system to initial state
        """
        self.state = np.copy(self.init_state)



    def evolve(self, unitary, save=False):
        """
        evolve the spin system 
        """

        # evolve state
        if save:
            self.state = unitary @ self.state
            return self.state
        else:
            return unitary @ self.state

    def measure(self, state=None, project=np.asarray([[1, 0]])):
        """
        perform projective measurement in computational basis
        """
        if state is None:
            state = self.state

        return np.abs(np.dot(project, state))**2

    def state_evolve(self, params, bloch=[False, 10], lite=False):
        """
        computes the probability of finding the system in the projection state over the given range
        """
        # ensure dictionary meets C structure expectations
        cparams = dict_sanitise(params)

        if lite:
            # run simulation using lite evolver (weird assignment errors if variable type changes from full version)
            
            self.probs = np.empty((1), dtype=np.float64)
            self.state_cache = np.empty((2), dtype=np.complex128)
            self.state_cache[:] = self.state
            quantumc.n_propagate_lite(cparams, self.state, self.state_cache, self.probs)
            return [params["tend"]], self.probs

        # simulate and return evolution data
        else:
            # time vector to evolve state over
            self.time = np.arange(cparams["tstart"],
                              cparams["tend"], cparams["dt"])
        

            # preallocate probability array
            self.probs = np.empty((len(self.time)), dtype=np.float64)
            # create list for snap shot states to plot
            if bloch:
                bloch_points = []
            # initialise state cache
            self.state_cache = np.empty((len(self.time), 2), dtype=np.complex128)
            # set initial state
            self.state_cache[0,:] = self.state
            quantumc.n_propagateN(self.time, cparams, self.state,
                                  self.state_cache, self.probs)

            if params["savef"]>1:
                self.probs = self.probs[::params["savef"]]
                self.time = self.time[::params["savef"]]

                    
            # plot evolution on the Bloch sphere
            if bloch[0]:
                for state in self.state_cache[::bloch[1]]:
                    state = np.matrix(state)
                    bloch_points.append(self.get_bloch_vec(np.outer(state.H, state)))


                # convert to qutips annoying format
                # TODO: preallocate
                x, y, z = [], [], []
                for vec in bloch_points:
                    x.append(vec[0])
                    y.append(vec[1])
                    z.append(vec[2])

                bloch_points = [x, y, z]
                return self.time, self.probs, bloch_points
            else:
                return self.time, self.probs


    def field_get(self,params):
        """
        Retrieves and returns the Hamiltonian field components
        given simulation definitions. 
        """
        # ensure simulation dictionary is compliant
        cparams = dict_sanitise(params)
        # compute time vector
        time = np.arange(cparams["tstart"],cparams["tend"],cparams["dt"])    
        # define bfield container
        Bfields = np.empty((len(time),3), dtype=float)
        # compute bfields
        quantumc.Bfield(time, cparams, Bfields)
        return time, cparams, Bfields


    def field_plot(self, params, bloch=[False,1]):
        """
        Plots the signal components given simulation parameters over the specified time doman
        """

        # get time varying fields and simulation data
        time,cparams,Bfields = field_get(params=params)

        # plot magnetic field vector on Bloch sphere
        if bloch[0]:
            Bfields = Bfields[::bloch[1],:]
            # normalise each magnetic field vector 
            for i in range(len(Bfields)):
                norm = np.sqrt(Bfields[i,0]**2 + Bfields[i,1]**2 + Bfields[i,2]**2) 
                Bfields[i,0] /= norm
                Bfields[i,1] /= norm
                Bfields[i,2] /= norm

            # extract x,y,z fields
            Bx = Bfields[:,0]
            By = Bfields[:,1]
            Bz = Bfields[:,2]


            # define bloch object
            b = Bloch() 
            b.add_points([Bx,By,Bz])
            b.show()

        else:
            # plot fields
            fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False)
            for i, row in enumerate(ax):
                # plot each component, skipping zero time value
                row.plot(time, Bfields[:,i])
                row.set_title("Field vector along {} axis".format(['x', 'y', 'z'][i]))
            
            plt.ylabel('Frequency (Hz)')
            plt.xlabel("Time (s)")
            #plt.ylabel("Ampltiude ($Hz/Gauss$)")
            plt.show()

    def frame_transform(self, cstate=None, frame=["interaction", "lab"], project=meas1["0"], bloch=[False, 10]):
        """
        EXPERIMENTAL
        Transform a state or set of states to a specified reference frame from another. This method
        is still in the experimental phase. It works well for going from simpler reference frames to
        complicated oned but the reverse is prone to numerical instability. 
        """
        if cstate is None:
            cstate = np.copy(self.state_cache)

        # compute reference frame map
        if callable(frame):
            unitary_map = frame
        # determine transition case and define appropriate time dependent unitary operator
        elif frame[0] == "lab":
            if frame[1] == "interaction":
                def unitary_map(t, larmor=gyro):
                    return np.asarray([[np.exp(1j*np.pi*larmor*t), 0], [0, np.exp(-1j*np.pi*larmor*t)]])
            elif frame == "dressed":
                # define dressed state transform
                def dressed(t, omega=1e4, detuning=0): return np.asarray([[np.cos(np.arctan(
                    omega/detuning)/2), -np.sin(np.arctan(omega/detuning)/2)], [np.sin(np.arctan(omega/detuning)/2), np.cos(np.arctan(omega/detuning)/2)]])

                def unitary_map(t, larmor=larmor): return dressed(t) @ np.asarray([[np.exp(1j*np.pi*larmor*t), 0], [0, np.exp(-1j*np.pi*larmor*t)]])
            else:
                raise ValueError("Unrecognised output reference frame")

        elif frame[0] == "interaction":
            if frame[1] == "lab":
                def unitary_map(t, larmor=gyro):
                    return np.asarray([[np.exp(-1j*np.pi*larmor*t), 0], [0, np.exp(1j*np.pi*larmor*t)]])
            elif frame == "dressed":
                # define dressed state transform
                def unitary_map(t, omega=1e4, detuning=0): return np.asarray([[np.cos(np.arctan(
                    omega/detuning)/2), np.sin(np.arctan(omega/detuning)/2)], [-np.sin(np.arctan(omega/detuning)/2), np.cos(np.arctan(omega/detuning)/2)]])
            else:
                raise ValueError("Unrecognised output reference frame")

        elif frame[0] == "dressed":
            if frame[1] == "interaction":
                def unitary_map(t, larmor=gyro):
                    return np.asarray([[np.cos(np.arctan(omega/detuning)/2), -np.sin(np.arctan(omega/detuning)/2)], [np.sin(np.arctan(omega/detuning)/2), np.cos(np.arctan(omega/detuning)/2)]])
            elif frame == "lab":
                # define dressed state transform
                def dressed(t, omega=1e4, detuning=0): return np.asarray([[np.cos(np.arctan(
                    omega/detuning)/2), -np.sin(np.arctan(omega/detuning)/2)], [np.sin(np.arctan(omega/detuning)/2), np.cos(np.arctan(omega/detuning)/2)]])

                def unitary_map(t, larmor=larmor): return dressed(t) @ np.asarray([[np.exp(1j*np.pi*larmor*t), 0], [0, np.exp(-1j*np.pi*larmor*t)]])
            else:
                raise ValueError("Unrecognised output reference frame")

        else:
            raise ValueError("Unrecognised input reference frame")

        #self.time = self.time[::100]
        #cstate = cstate[::100,:]
        # apply transform to states
        if len(np.shape(cstate)) == 2:
            new_states = [unitary_map(t) @ cstate[step, :] for step, t in enumerate(self.time)]
            nprobs = np.squeeze([np.abs(project @ nstate)**2 for i, nstate in enumerate(new_states)])
            # save new states and projection probabilities
            self.state_cache = new_states
            self.probs = nprobs
        else:
            new_states = unitary_map(self.time[-1]) @ cstate
            return new_states

    def bloch_plot(self, points=None):
        """
        Plot the current state on the Bloch sphere using
        qutip. 
        """
        if points is None:
            # convert current state into density operator
            rho = np.outer(self.state.H, self.state)
            # get Bloch vector representation
            points = self.get_bloch_vec(rho)
            # Can only plot systems of dimension 2 at this time
            assert len(
                points) == 3, "System dimension must be spin 1/2 for Bloch sphere plot"

        # create instance of 3d plot
        bloch = Bloch(figsize=[9,9])
        # add state
        bloch.add_points(points)
        bloch.add_vectors([0,0,1]) 
        bloch.render(bloch.fig, bloch.axes)
        bloch.fig.savefig("bloch.png",dpi=600, transparent=True)
        bloch.show()



    def get_bloch_vec(self, rho):
        """
        compute the bloch vector for some 2 dimensional density operator rho
        """
        u = 2*np.real(rho[0, 1])
        v = 2*np.imag(rho[1, 0])
        w = np.real(rho[0, 0] - rho[1, 1])
        return [u, v, w]

    def exp_plot(self, time, probs, title="Expectation value over time"):
        """
        Formatted code for plot (why must plot code always be hideous?)
        """
        # compute expectation values
        Fe = [2*p-1 for p in probs]
        plt.figure(figsize=[12,8])
        plt.plot(time, Fe)
        plt.ylim([-1.05, 1.05])
        plt.xlim([time[0], time[-1]])
        plt.grid()
        plt.xlabel("Time (s)")
        plt.ylabel("Expectation Value")
        plt.title(title)
        plt.show()

    def bloch_animate(self, pnts, name="Bloch_animate"):
        """
        Animates the path of a state through the set of pure states - requires ffmpeg
        """
        from pylab import figure
        import matplotlib.animation as animation
        from mpl_toolkits.mplot3d import Axes3D

        # set up plot environment
        fig = figure()
        ax = Axes3D(fig, azim=-40, elev=30)
        sphere = Bloch(axes=ax)

        # define animation function (from qutip docs)
        def animate(i):
            sphere.clear()
            sphere.add_points([pnts[0][:i+1], pnts[1][:i+1], pnts[2][:i+1]])
            sphere.make_sphere()
            return ax

        def init():
            sphere.vector_color = ['r']
            return ax

        ani = animation.FuncAnimation(fig, animate, np.arange(len(pnts[0])),
                                      init_func=init, repeat=False)

        ani.save(name + ".mp4", fps=20)


def recon_pulse(sim_vars, plot=True, savefig=False):

    bfreqs, projs, sim_def = atomic_sense(**sim_vars)

    # add to simulation parameter dictionary if not already defined
    if "measurements" not in sim_vars.keys():
      sim_vars["measurements"] = 50

    # normally distributed noise if requested
    if not np.isclose(sim_vars["noise"], 0.0):
      projs += np.random.normal(loc=0.0, scale=sim_vars["noise"], size=np.size(projs))

    # determine time vector and add to variable dictionary 
    time = np.arange(sim_def["tstart"], sim_def["dett"], 1/sim_vars["fs"])
    sim_vars["time"] = time

    # generate measurement transform
    transform, ifreqs, _ = measure_gen(ndim=len(time), 
                                    time=time, 
                                    basis="fourier", 
                                    measurements=sim_vars["measurements"], 
                                    freqs=bfreqs)
    

    ifreqs = np.asarray(ifreqs)

    # compute measurement record, selecting only those chosen for reconstruction
    meas_record = projs[[i for i in ifreqs]] 
    # remove probability bias (looks like there is a DC component otherwise)
    bias = np.mean(meas_record)
    meas_record -= bias

    # perform single event matched filtering
    if sim_vars["method"]=="match":
      # define template
      exit()
    # perform multievent matched filtering
    elif sim_vars["method"]=="mulmatch":
      # generate matching template 
      t = np.arange(0,1/sim_vars["sig_freq"],1/sim_vars["fs"])
      template = np.sin(2*pi*sim_vars["sig_freq"]*t)
      
      # create optimiser instance
      comp = CAOptimise(svector=meas_record,   # measurement vector in sensor domain
                      transform=transform,   # sensor map from source to sensor
                      template=template,
                      verbose=True,          # give me all the information
                      **sim_vars)           # epsilon and other optional stuff

      # generate actual neural signal for comparison purposes
      signal = pulse_gen(sim_vars["sig_freq"], tau=[sim_vars["tau"]], amp=sim_vars["sig_amp"])(time)

      # perform matched filtering identification
      comp.py_notch_match(osignal=signal)
      
      # extract reconstruction information
      recon = comp.template_recon
      notch = comp.notch

      # format reconstruction information and save
      print("Storing signal_reconstruction")
      recon_sig = np.empty((4,len(notch)), dtype=np.float)
      recon_sig[0,:] = time
      recon_sig[1,:] = comp.correlation
      recon_sig[2,:] = recon
      recon_sig[3,:] = notch
      data_store(sim_vars, recon_sig, root_group="Signal_Reconstructions", verbose=True)

    else:
      if sim_vars["method"] != "default":
        print("Unrecognised reconstruction method, using default")
        sim_vars["method"] = "default"
      # create optimiser instance
      comp = CAOptimise(svector=meas_record,   # measurement vector in sensor domain
                      transform=transform,   # sensor map from source to sensor
                      verbose=True,          # give me all the information
                      **sim_vars)           # epsilon and other optional stuff

      # reconstruct signal
      comp.cvx_recon()
      # extract signal estimate
      recon = comp.u_recon

      # save to archive
      recon_sig = np.empty((2,len(recon)), dtype=np.float)
      recon_sig[0,:] = time
      recon_sig[1,:] = recon
      data_store(sim_vars, recon_sig, root_group="Signal_Reconstructions", verbose=True)

          
    if plot:
      # measurement frequencies used
      comp_f = bfreqs[ifreqs]
      # measurement record adjustment
      comp_p = meas_record + bias
      recon[-20:] = 0
      # plot reconstruction
      plot_gen_2(bfreqs, projs, comp_f, comp_p, time, recon, {**sim_def,**sim_vars}, savefig=savefig)

    return bfreqs, projs



def field_gen(field_params):
    """
    Creates a 3 element list of functions that wholly defines a classical electromagnetic field 
    given a dictionary of parameters
    """

    # list of field functions
    field_vector = []
    for i, struct in enumerate(field_params["struct"]):

        # generate a simple sinusoid function with amplitude and frequency
        if struct is "sinusoid":
            field_vector.append(
                lambda t, j=i: field_params["amp"][j]*(np.cos(2*np.pi*field_params["freqb"][j]*t)))
        # generate a constant DC bias field
        elif struct is "constant":
            field_vector.append(lambda t, j=i: field_params["amp"][j]*t/t)
        # generate a pulsed sinusoid with frequency omega beginning at time tau (seconds)
        elif struct is "pulse":
            # generate field callable with num pulsed sinusoids
            field_vector.append(pulse_gen(
                field_params["freqb"][i], field_params["tau"][i], amp=field_params["amp"][i]))
        elif struct is "tchirp":
            # define chirp component
            chirp = lambda t, j=i: np.heaviside(t-field_params["tau"][j], 1.0)*field_params["misc"][j]*np.tanh(
                0.01*(t-field_params["tau"][j])/field_params["tau"][j])
            # generate field with time varying amplitude
            constant = lambda t, j=i: field_params["amp"][j]
            # add to field vectors
            field_vector.append(lambda t: constant(t) + chirp(t))
        elif struct is "custom":
            field_vector.append(field_params["misc"][i])
        else:
            raise ValueError("Unrecognised field type: {}".format(struct))

    return field_vector



class Hamiltonian(object):
    """
    Defines the Hamiltonian that acts on a SpinSystem class - dimension of Hilbert spaces of the two must match
    """

    def __init__(self, spin="half",  freq=500):
        # spin of system
        self.spin = spin
        # define oscillation frequency
        self.freq = freq
        # define energy seperation
        self.energy = hbar*self.freq
        # define internal Hamiltonian for spin system
        if self.spin == "half":
            # two energy eigenstates (0,1)
            self.free_ham = 0.5*hbar * \
                np.asarray([[-self.freq, 0], [0, self.freq]],
                           dtype=np.complex128)
        elif self.spin == "one":
            # TODO: three energy eigenstates (0,1,2)
            pass
        else:
            print("Unrecognised spin system {}: aborting".format(self.spin))

    def generate_simple(self, potential=None,):
        """
        Creates a simple Hamiltonian based off the input Potential or defaults
        """
        # potential component of Hamiltonian
        self.potential = potential

        # check if potential is custom or one of defaults of style [coupling, phase, rwa]
        if self.potential is not None:
            if type(self.potential) == list:
                coupling = self.potential[0]
                tuning = self.potential[1]

            elif self.init is "zero":
                self.state = op1["pz"]

                # apply rotating wave approximation
                if potential[2]:
                    self.potential = 0.0
                    self.hamiltonian = 0.5*hbar*np.asarray([[tuning, coupling],
                                                            [coupling, -tuning]], dtype=np.complex128)
                # full hamiltonian in interaction picture (t dependency is annoying)
                else:
                    # define the off diagonal components as lambda functions for t dependency
                    def offdiag(t): return coupling * \
                        (1 + np.exp(-2j*(self.freq+tuning)*t))
                    self.hamiltonian = lambda t: 0.5*hbar*np.asarray([[tuning, offdiag(t)],
                                                                      [np.conj(offdiag(t)), -tuning]],
                                                                     dtype=np.complex128)
            else:
                pass
        else:
            self.potential = 0.0

    def generate_field_hamiltonian(self, fields):
        """
        generates an arbitrary magnetic field hamiltonian. Field must
        be a function with time as its only argument.
        """
        # store field
        self.fields = fields
        # enforce function type
        for field in fields:
            assert callable(field), "Field {} must be callable".format(i)
        # redundant constants for clarity
        self.hamiltonian = lambda t: 0.5*hbar*2*np.pi * \
            (fields[0](t)*op1["x"] + fields[1](t)
             * op1["y"] + fields[2](t)*op1["z"])

        # cache version of hamiltonian
        def hamiltonian2(time):
            return 0.5*hbar*2*np.pi*np.asarray([[fields[2](time), fields[0](time)+1j*fields[1](time)],
                                                [fields[0](time)-1j*fields[1](time), -1*fields[2](time)]])
        self.hamiltonian_cache = hamiltonian2

    def bloch_field(self, time):
        """
        Plot magnetic field vector normalised to the bloch sphere
        """
        # seperate coordinates into axis set
        xb = self.fields[0](time)
        yb = self.fields[1](time)
        zb = self.fields[2](time)
        # add to list and normalise
        points = [list(xb), list(yb), list(zb)]
        sqsum = 0.0
        for i, point in enumerate(points[0]):
            # compute magnitude and store largest value
            sqsum_test = np.sqrt(
                points[0][i]**2 + points[1][i]**2 + points[2][i]**2)
            if sqsum_test > sqsum:
                sqsum = sqsum_test

        points = points/sqsum

        # create Bloch sphere instance
        bloch = Bloch()
        # add points and show
        bloch.add_points(points)
        bloch.show()


def data_compare(a,b):
    """
    Compares two variables and checks for equivalence.
    """

    # iterate through lists 
    if (type(a) == list) or (type(a)==tuple):
        for i,j in zip(a,b):
            equal = data_compare(i,j)
            if not equal:
                return False
    # valid for booleans, floats, ints and complex numbers
    if np.isclose(a,b):
        return True
    # default to False (safer)
    return False


def data_retrieval(sim_params, ignore=[], root_group = "Atomic_Simulations",verbose=True):
    """
    Retrieves a data set from the archive if it has already been simulated with identical parameters
    """

    # compute hash
    hash_name = str(dict2hash(sim_params, ignore=["proj"]))

    # open archive and check if atomic sensor data exists
    with h5py.File(archivepath, 'a') as archive:
        # create group if it doesn't exist
        if root_group not in archive:
            atomic_sense = archive.create_group(root_group)
            if verbose:
                print("No matching result in archive")
            return None
        else:
            atomic_sense = archive[root_group]

        # check if parameter hash is in group
        if hash_name in atomic_sense:
            # dataset matches
            if verbose:
                print("Matching result found in archive")
            return np.asarray(atomic_sense[hash_name])            

    return None

def dict2hash(params, ignore=[]):
    """
    computes the hash of an input <params> while ignoring keys defined
    in <ignore>. Poor mans hash table
    """
    # always use the same initial seed

    hash_str = ''
    # iterate through dictionary
    for key,val in params.items():
        # skip keys in ignore list
        if key in ignore: continue

        # round ot 5 decimal places to avoid any precision errors
        if isinstance(val, (float, np.float, np.float64)):
            val = np.round(val, decimals=5)

        # add str value to dict string
        hash_str += str(val)
    # compute hash
    hash_obj = blake2b()
    hash_obj.update(hash_str.encode("UTF-8"))
    hash_val = hash_obj.digest()

    return hash_val



def data_store(params, data, root_group="Atomic_Simulations",verbose=True):
    """
    Stores a simulation instance.
    """

    # compute hash name
    root = str(dict2hash(params, ignore=["proj"]))
    # turn parameter set into name string
    #date = gmtime()
    #root = "{}_{}_{}_{}_{}_{}".format(date.tm_year,date.tm_mon, date.tm_mday, date.tm_hour, date.tm_min, date.tm_sec)


    # open archive and check if atomic sensor data exists
    with h5py.File(archivepath, 'a') as archive:
        # create group if it doesn't exist
        if root_group not in archive:
            atomic_s = archive.create_group(root_group)
        else:
            atomic_s = archive[root_group]

        if verbose:
            print("Saving simulation results to archive file")
        try:
            dataset = atomic_s.create_dataset(root, data=np.asarray(data))

            # save attributes
            for key, val in params.items():
                dataset.attrs[key] = val
        except RuntimeError:
            if verbose:
                print("Reconstruction instance already exists")           


def atomic_sense(sig_amp=100, tau=1e-4, sig_freq=1e4, f_range=[250, 450, 1], parallel=True, savef=1000, verbose=True, **args):
    """
    Simulate the atomic sensor.
    """

    # default simulation parameters
    params = {"tstart":      0.0,               # time range to simulate over
              "tend":        0.5, 
              "dt":         1e-8,
              "larmor":     gyro,               # bias frequency (Hz)
              "phi":        0.0,                # phase shift (rad)
              "rabi":       1000,               # dressing amplitude (Hz/2)
              "rff":        gyro,               # dressing frequency (Hz)
              "rph":        0,                  # dressing phase
              "nf":         1e4,                # neural signal frequency (Hz)
              "dett":       0.1,                # detuning sweep start time
              "detA":       2000,               # detuning amplitude
              "beta":        40,                 # detuning temporal scaling
              "dete":      0.495,               # detuning truncation time
              "xlamp":       100,               # line noise amplitude
              "xlfreq":       50,                # line noise frequency              
              "xlphase":     0.1,                # start phase
              "zlamp":         1,
              "zlfreq":     1000,
              "zlphase":     0.0,
              "proj": meas1["0"]}               # measurement projector
    

    # set signal params to be compliant with C extension
    params["sA"] = sig_amp       
    params["nt"] = tau
    params["nte"] = tau+1/sig_freq
    params["nf"] = sig_freq



    # change values from defaults if set
    for key,val in args.items():
        if key in params.keys():
            params[key] = val

    # projection array
    projs = []
    # frequencies to sample
    freqs = np.arange(f_range[0], f_range[1], f_range[2])

    # iterate simulation over dressing frequencies
    for freq in freqs:
        if verbose:
            print("Computing evolution with tuning frequency: {:.2f} Hz\r".format(
                freq), end="", flush=True)

        # set tuning frequency
        params["rabi"] = freq

        # adjust for Bloch-Siegert shift
        shift = (params["rabi"]**2)/(4*params["rff"]) + (params["rabi"]**4/(4*(params["rff"])**3))
        params["larmor"] = gyro + shift

        # set small random sigma_z phase shift
        params["xlphase"] = np.round((-2*np.random.random() + 1)*np.pi/8, decimals=3)

        ### check if simulation has been run here using params ###
        result = data_retrieval(params, ignore=['proj'], verbose=False)

        # result not found
        if result is None:
            # redefine atom spin system
            atom = SpinSystem(init="zero")
            # evolve state using hamiltonian without saving intermediate states
            tdomain, probs = atom.state_evolve(params=params, lite=True)
            # format into single chunk for hdf5 compression
            data_array = np.zeros((2, len(probs[::-savef])))
            # decimate data according to save frequency
            data_array[0, :] = tdomain[::-savef]
            data_array[1, :] = probs[::-savef]
            # store data in archive
            data_store(params, data_array)   
        else:
            tdomain = result[0,:]
            probs = result[1,:]

        # add output probability
        projs.append(probs[-1])

    return np.asarray(freqs), np.asarray(projs), params



## DEFUNCT WITH NEW SIMULATION SCHEME ##
def _pseudo_fourier(f_range=[250, 450, 1], noise=0.0, plot=False, verbose=True):
    """
    Computes the projection |<1|psi>|^2 of a two level system for a given point in the signal parameter space with different frequency tunings
    """
    result = None  # data_retrieval(sim_params)
    if result is None:
        # projection array
        projs = []
        # frequencies to sample
        freqs = np.arange(f_range[0], f_range[1], f_range[2])
        # initiate Hamiltonian instance
        ham = Hamiltonian()
        for freq in freqs:
            if verbose:
                print("Computing evolution with tuning frequency: {:.2f} Hz\r".format(
                    freq), end="", flush=True)

            def Bx(t, omega_amp=freq): return omega_amp*(t/t)

            def By(t): return 0

            def Bz(t): return pulse(t)

            # define magnetic field vector parameters
            params = {"struct": ["custom", "constant", "custom"],
                      "freqb": [sig_freq, 50, 0],          # frequency in Hz
                      "tau": [tau, None, None],          # time event of pulse
                      # amplitude in Gauss -> 1 Gauss ~= 700000 Hz precession
                      "amp": [sig_amp, 0, freq],
                      "misc": [Bz, None, Bx]}          # misc parameters
            # generate magentic fields
            fields = field_gen(field_params=params)
            # compute Hamiltonian for updated magnetic field
            ham.generate_field_hamiltonian(fields)
            # redefine atom spin system
            atom = SpinSystem(init="super")
            # evolve state using hamiltonian
            time, probs, pnts = atom.state_evolve(
                t=[1e-44, t, 1/2e5], hamiltonian=ham.hamiltonian_cache, project=meas1["1"], bloch=[False, 5])
            # atom.frame_transform(project=meas1["0"])
            projs.append(probs[-1])

        projs = np.asarray(projs)
        # format into single chunk for hdf5 compression
        data_matrix = np.zeros((2, len(projs)))
        signal_matrix = np.zeros((4, len(time)))
        data_matrix[0, :] = freqs
        data_matrix[1, :] = projs
        signal_matrix[0, :] = time
        signal_matrix[1, :] = fields[0](time)
        signal_matrix[2, :] = fields[1](time)
        signal_matrix[3, :] = fields[2](time)

        sim_params["name"] = "Fourier"
        #data_store(sim_params, data=data_matrix, name="_Fourier_measurements")
        sim_params["name"] = "Field"
        #data_store(sim_params, data=data_matrix, name="_Field_signal")

    else:
        freqs = result[0, :]
        projs = result[1, :]
        # define magnetic field vector parameters
        params = {"struct": [struct, "constant", "constant"],
                  "freqb": [sig_freq, 0, 0],          # frequency in Hz
                  "tau": [tau, None, None],          # time event of oulse
                  # amplitude in Gauss -> 1 Gauss ~= 700000 Hz precession
                  "amp": [sig_amp, 0, 0],
                  "misc": [sig, None, None]}          # misc parameters
        # generate magentic fields
        fields = field_gen(field_params=params)

    # set max of uniform distribution to be max of projector oscillations
    #noise_lvl = noise*np.max(projs-np.mean(projs))
    # add white noise to projector results
    #projs += np.random.normal(scale=noise_lvl,size=np.size(projs))
    if plot:
        plt.plot(freqs, projs, 'o--')
        plt.xlabel("Frequency of Bias field (Hz)")
        plt.ylabel("$|<1|\psi(t_1)>|^2$")
        plt.title("Probability vs tuning frequency for {} Hz {} beginning at {} seconds".format(
            sig_freq, params["struct"][0], params["tau"][0]))
        plt.grid()
        plt.show()

    return np.asarray(freqs), np.asarray(projs), fields[0]

def demonstrate(in_vars):
    # set random seed for tuning frequency choice
    np.random.seed(141)
    # define user parameters for simulation run
    sim_vars = {"measurements":        50,           # number of measurements
                "epsilon":           0.01,           # radius of hypersphere in measurement domain
                "sig_amp":             40,           # amplitude of magnetic field signal in Gauss/Hz
                "sig_freq":          5023,           # frequency of magnetic field signal
                "tau":              0.033,           # time events of pulses
                "f_range":  [4800,5200,5],            # frequency tunings of BECs
                "noise":             0.00,           # noise to add to measurement record SNR percentage e.g 0.1: SNR = 10
                "zlamp":                0,
                "method":       "default",
                "savef":             5000,
                "fs":               2.2e4}

    for key in sim_vars.keys():
        if key in in_vars.keys():
            sim_vars[key] = in_vars[key]

    bfreqs1, projs1 = recon_pulse(sim_vars, plot=True, savefig=False)
    

