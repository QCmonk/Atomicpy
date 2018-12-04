cimport cython
import numpy as np
from cython.parallel import prange
from libc.math cimport sqrt, abs, sin, cos, M_PI, exp, ceil


# define dictionary parameter structure for defining problem
ctypedef struct Params:
    # simulation time
    double tstart
    double tend
    double dt

    # dress field parameters
    double phi
    double rabi
    double rff
    double rph

    #bias field parameters
    double larmor

    # neural signal parameters
    double nf
    double sA
    double nt
    double nte 

    # detuning parameters
    double dett # detuning time start
    double beta # detuning sweep parameter
    double detA # detuning max amplitude
    double dete # detuning truncation time

    # sigma_x AC line noise parameters
    double xlamp   # amplitude
    double xlfreq  # frequency 
    double xlphase # phase

    # sigma_z AC line noise parameter
    double zlamp
    double zlfreq
    double zlphase

    # measurement projector
    double complex proj[2]
    

# import complex casting function from complex library
cdef extern from "complex.h":
    double complex CMPLX(double, double)


# define hyperbolic functions
@cython.cdivision(True)
cdef inline double tanh(double x):
    cdef double tx = 2*x
    return (exp(tx)-1)/(exp(tx)+1)

@cython.cdivision(True)
cdef inline double sech(double x):
    return 2/(exp(x)+exp(-x))





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def n_propagate_lite(paramdict, double complex [:] initial_state, double complex [:] state_out, double [:] probs):
    """
    State evolver without state saving, much easier on memory requirements. U
    Uses second order unitary approximation.
    """

    # inialise C-type dictionary -> structure
    cdef Params params
    params = paramdict

    # de

    # define loop constants
    cdef int i 
    cdef unsigned long int steps = <int>ceil((params.tend-params.tstart)/params.dt)
    cdef double time = params.tstart
    cdef double complex[:] state = initial_state
    state_out[:] = state

    # compute initial probability
    probs[0] = absquare(params.proj[0]*state[0] + params.proj[1]*state[1])

    params.dt = params.dt*0.5

    # iterate through time using second order expansion
    for i in range(1,steps):
        time += 2*params.dt
        # apply half step unitary at current position
        unitary22(time, &params, state[:], state_out[:], &probs[0])
        # apply half step at destination time
        unitary22(time + 2*params.dt, &params, state[:], state_out[:], &probs[0])



@cython.boundscheck(False) # deactivate index bound checks locally
@cython.wraparound(False)  # deactivate index wraparound check
@cython.nonecheck(False)   # deactivate none checks (these will segfault if you pass a None-type)
def n_propagateN(double [:] time, paramdict, double complex [:] initial_state, double complex [:,:] state_out, double [:] probs):
    """
    Fast unitary evolution over <time> given simulation parameters <paramdict>.
    """
    
    # inialise C-type dictionary -> structure
    cdef Params params
    params = paramdict

    # define loop constants
    cdef int i 
    cdef int steps = len(time)
    cdef double complex[:] state = initial_state
    state_out[0,:] = state
    probs[0] = absquare(params.proj[0]*state[0] + params.proj[1]*state[1])

    # set time step to half
    params.dt = params.dt*0.5

    # iterate through time using second order expansion
    for i in range(1,steps):
        # apply half step unitary at current position
        unitary22(time[i], &params, state[:], state_out[i,:], &probs[i])
        # apply half step at destination time
        unitary22(time[i] + 2*params.dt, &params, state[:], state_out[i,:], &probs[i])





@cython.boundscheck(False) # deactivate index bound checks locally
@cython.wraparound(False)  # deactivate index wraparound check
@cython.nonecheck(False)   # deactivate none checks (these will segfault if you pass a None-type)
def n_propagate(double [:] time, paramdict, complex [:] initial_state, complex [:,:] state_out, double [:] probs):
    """
    Fast unitary evolution over <time> given simulation parameters <paramdict>.
    """
    # inialise C-type dictionary -> structure
    cdef Params params
    params = paramdict

    # define loop constants
    cdef int i 
    cdef int steps = len(time)
    cdef double complex[:] state = initial_state
    state_out[0,:] = state
    probs[0] = absquare(params.proj[0]*state[0] + params.proj[1]*state[1])

    # iterate through time
    for i in range(1,steps):
        unitary22(time[i], &params, state[:], state_out[i,:], &probs[i])

@cython.boundscheck(False) # deactivate index bound checks locally
@cython.wraparound(False)  # deactivate index wraparound check
@cython.nonecheck(False)   # deactivate none checks (these will segfault if you pass a None-type)
@cython.cdivision(True)
# computes an updated state given an input and the 3 B field components
cdef inline void unitary22(double t, Params *params, double complex [:] state_in, double complex [:] state_out, double *probs):

    # define intermediate variables
    cdef double norma,normb, E, a, b, c
    cdef double complex phase, psi_up, psi_down
    cdef double complex U11, U22, U12, U21, sa, sb

    # compute "hamiltonian" for time t
    angles(t, params, &a, &b, &c)


    # can't handle a==b==0 with cdivision activated
    if abs(a)<1e-100 and abs(b)<1e-100:
        a=1e-100

    # compute energy
    E = sqrt(a*a + b*b + c*c)

    # compute eigenvalue while still local (definitely don't want trig functions to wait on L3)
    phase = cos(E) + 1j*sin(E)

    # compute unnormalised eigenvectors (force casting after some strange assignment glitch)
    U11 = <complex>(c - E)
    U12 = <complex>(c + E)
    U21 = a + 1j * b
    U22 = U21

    # compute unitary by applying decomposition to state piecewise
    psi_down = U11.conjugate()*state_in[0] + U21.conjugate()*state_in[1]
    psi_up = U12.conjugate()*state_in[0] + U22.conjugate()*state_in[1]

    # apply phase factor
    psi_down *= phase
    psi_up *= phase.conjugate()

    # compute squared normalisation
    norma = absquare(U11) + absquare(U21)
    normb = absquare(U12) + absquare(U22)

    # apply final eigenvector and save to output state memory location
    state_out[0] = U11*psi_down/norma + U12*psi_up/normb
    state_out[1] = U21*psi_down/norma + U22*psi_up/normb

    # update initial state
    state_in[0] = state_out[0]
    state_in[1] = state_out[1]

    # measurement probability given projector
    probs[0] = absquare(params.proj[0]*state_out[0] + params.proj[1]*state_out[1])


# compute the absolute square of a complex number
cdef inline double absquare(double complex z):
    return z.real*z.real + z.imag*z.imag


# computes the B field angles given a time step and an input array
cdef inline void angles(double t, Params *params, double *a, double *b, double *c):
    # define omega, detune, wrap
    cdef double omegaD, detune
    
    # check if detuning run needs to be applied
    if t < params.dett:
        # define generic fields using structure constants  
        a[0] = 2*M_PI*(params.rabi*cos(2*M_PI*(params.rff)*t) + params.xlamp*sin(2*M_PI*params.xlfreq*t + params.xlphase) )*params.dt
        b[0] = 0.0

        # add neural impulse
        if t<params.nt or t>params.nte:
           c[0] = M_PI*(params.larmor + params.zlamp*sin(2*M_PI*params.zlfreq*t + params.zlphase))*params.dt
        else:
           c[0] = M_PI*(params.larmor + params.sA*sin(2*M_PI*params.nf*(t-params.nt))  + params.zlamp*sin(2*M_PI*params.zlfreq*t + params.zlphase))*params.dt


    elif t < params.dete:
        
        omegaD = params.rabi*sech(params.beta*(t-params.dett))
        detune = params.detA*tanh(params.beta*(t-params.dett))

        # a[0] = 2*M_PI*(omegaD*cos(2*M_PI*(params.rff+detune)*(t-params.dett)) + params.xlamp*sin(2*M_PI*params.xlfreq*t + params.xlphase))*params.dt
        # b[0] = 0.0 
        # c[0] = M_PI*(params.larmor + params.zlamp*sin(2*M_PI*params.zlfreq*t + params.zlphase))*params.dt

        a[0] = 2*M_PI*(omegaD*cos(2*M_PI*(params.rff+detune)*(t-params.dett)))
        b[0] = 0.0 
        c[0] = M_PI*(params.larmor)*params.dt


    else:

        # detuning truncation
        a[0] = params.xlamp*sin(2*M_PI*params.xlfreq*t + params.xlphase)*params.dt
        b[0] = 0.0
        c[0] = M_PI*(params.larmor + params.zlamp*sin(2*M_PI*params.zlfreq*t + params.zlphase))*params.dt

@cython.boundscheck(False) # deactivate index bound checks locally
@cython.wraparound(False)  # deactivate index wraparound check
@cython.nonecheck(False)   # deactivate none checks (these will segfault if you pass a None-type)
def Bfield(double [:] time, paramdict, double [:,:] Bfield):
    """
    Computes angles over the specifed time range
    """
    # inialise C-type dictionary -> structure
    cdef Params params
    params = paramdict

    # define loop constants
    cdef int i 
    cdef int steps = len(time)

    # iterate through time
    for i in range(0,steps):
        angles(time[i], &params, &Bfield[i,0], &Bfield[i,1], &Bfield[i,2])


