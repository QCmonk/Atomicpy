# cython: profile=False
cimport cython
import numpy as np
from cython.parallel import prange
from libc.math cimport sqrt, abs, sin, cos, M_PI, exp, ceil,  atan2, acos#, sincos
from scipy.linalg import expm


cdef double sqrt2 = 1.41421356237309504880168872420969807856967187537694
cdef double invsqrt2 = 0.70710678118654752440084436210484903928483593768847
cdef double eps = np.finfo(np.float64).eps*100
cdef double M_TAU = 2*M_PI
cdef double M_SQRT3 = 1.73205080756887729352744634151 #sqrt(3)

# compute the absolute square of a complex number
@cython.profile(False)
cdef inline double absquare(double complex z):
    return z.real*z.real + z.imag*z.imag

cdef inline double square(double z):
    return z*z


# define dictionary parameter structure for defining problem
ctypedef struct Params:
    # simulation time
    double tstart
    double tend
    double dt
    int savef

    # dress field parameters
    double phi
    double rabi
    double rff
    double rph

    #bias field parameters
    double larmor

    double quad

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
    double tr2

    # sigma_z AC line noise parameter
    double zlamp
    double zlfreq
    double zlphase

    # measurement projector
    double complex proj[3]

    #storage parameters for fast field calculation
    double cdtrabi #2*Pi*Rabi*cos(2*Pi*rff*dt) - constant over simulation
    double sdtrabi #2*Pi*Rabi*sin(2*Pi*rff*dt) - constant over simulation
    double ctcrabi #cos(2*Pi*rff*t) - where t is the 'current simulation' time. This changes over simulation
    double stcrabi #sin(2*Pi*rff*t) - where t is the 'current simulation' time. This changes over simulation

    double cdtxl #cos(2*Pi*rff*dt) - constant over simulation
    double sdtxl #sin(2*Pi*rff*dt) - constant over simulation
    double ctcxl #cos(2*Pi*rff*t) - where t is the 'current simulation' time. This changes over simulation
    double stcxl #sin(2*Pi*rff*t) - where t is the 'current simulation' time. This changes over simulation


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
@cython.cdivision(True)
def n_propagate_lite(paramdict, double complex [:] state_in, spin):
    """
    Fast unitary evolution over <time> given simulation parameters <paramdict>.
    """
    # inialise C-type dictionary -> structure
    cdef Params params
    params = paramdict
    cdef int dim
    if spin == 'half':
        dim = 2
    else: #spin == 1
        dim = 3

    # define loop constants
    cdef int i, j
    cdef unsigned long int steps = <int>ceil((params.tend-params.tstart)/params.dt)
    
    #Initialise state array
    cdef double complex state[3]

    for j in range(0,dim):
        state[j] = state_in[j]

    #Intialise cos and sin values for fast field calculations
    params.cdtrabi = cos(M_TAU*params.rff*params.dt)#2*Pi*Rabi*cos(2*Pi*rff*dt) - constant over simulation
    params.sdtrabi = sin(M_TAU*params.rff*params.dt) #2*Pi*Rabi*sin(2*Pi*rff*dt) - constant over simulation
    params.ctcrabi = cos(M_TAU*params.rff*params.tstart) #cos(2*Pi*rff*t) - where t is the 'current simulation' time. This changes over simulation
    params.stcrabi = sin(M_TAU*params.rff*params.tstart)  #sin(2*Pi*rff*t) - where t is the 'current simulation' time. This changes over simulation

    # set time step to half
    cdef double t = params.tstart
    params.dt *= 0.5

    cdef double B[4]
    B[0] = M_TAU*params.rabi*cos(M_TAU*params.rabi*params.tstart)
    B[1] = 0.0
    B[2] = M_TAU*params.larmor
    B[3] = M_TAU*params.quad

    cdef double complex U[3][3] #unitary operator for spin 1

    import mytime

    if spin == 'half':
        # iterate through time using second order expansion
        for i in range(1,steps):
            # apply half step unitary at current position
            unitary22(params.dt, state, B)

            t += 2*params.dt
            fastB(t, &params, B)

            # apply half step at destination time
            unitary22(params.dt, state, B)

    elif spin == 'one': 
        # iterate through time using second order expansion
        unitary33(params.dt, B, U)
        for i in range(1,steps):
            # apply half step unitary at current position
            updateState33(state, U)
            #unitary33(time[i], &params, state, state)
 
            t += 2*params.dt
            fastB(t, &params, B)
            unitary33(params.dt, B, U)
            updateState33(state, U)

    state_in[0] = state[0]
    state_in[1] = state[1]
    if dim == 3:
        state_in[2] = state[2]

@cython.boundscheck(False) # deactivate index bound checks locally
@cython.wraparound(False)  # deactivate index wraparound check
@cython.nonecheck(False)   # deactivate none checks (these will segfault if you pass a None-type)
def n_propagateN(paramdict, double complex [:,:] states, double [:] probs, spin):
    """
    Fast unitary evolution over <time> given simulation parameters <paramdict>.
    """
    import mytime
    # inialise C-type dictionary -> structure
    cdef Params params
    params = paramdict
    cdef int dim
    if spin == 'half':
        dim = 2
    else: #spin == 1
        dim = 3

    # define loop constants
    cdef int i, j
    cdef unsigned long int steps = <int>ceil((params.tend-params.tstart)/params.dt)
    
    #Initialise state array
    cdef double complex state[3] #= <double complex *>malloc(dim * sizeof(double complex))

    for j in range(0,dim):
        state[j] = states[0][j]
    #1st time step to propagate


    #Intialise cos and sin values for fast field calculations
    params.cdtrabi = cos(M_TAU*params.rff*params.dt)#2*Pi*Rabi*cos(2*Pi*rff*dt) - constant over simulation
    params.sdtrabi = sin(M_TAU*params.rff*params.dt) #2*Pi*Rabi*sin(2*Pi*rff*dt) - constant over simulation
    params.ctcrabi = cos(M_TAU*params.rff*params.tstart) #cos(2*Pi*rff*t) - where t is the 'current simulation' time. This changes over simulation
    params.stcrabi = sin(M_TAU*params.rff*params.tstart)  #sin(2*Pi*rff*t) - where t is the 'current simulation' time. This changes over simulation

    # set time step to half
    cdef double t = params.tstart
    params.dt = params.dt*0.5

    cdef double B[4]
    B[0] = M_TAU*params.rabi*cos(M_TAU*params.rabi*params.tstart)
    B[1] = 0.0
    B[2] = M_TAU*params.larmor
    B[3] = M_TAU*params.quad

    cdef double complex U[3][3] #unitary operator for spin 1

    if spin == 'half':
        # compute initial probability
        probs[0] = absquare(params.proj[0].conjugate()*state[0] + params.proj[1].conjugate()*state[1])

        # iterate through time using second order expansion
        for i in range(1,steps):
            # apply half step unitary at current position
            unitary22(params.dt, state, B)

            t += 2*params.dt
            fastB(t, &params, B)

            # apply half step at destination time
            unitary22(params.dt, state, B)

            #store state and store probability along specified projection
            if i % params.savef == 0:
                states[i/params.savef][0] = state[0]
                states[i/params.savef][1] = state[1]

                probs[i/params.savef] = absquare(params.proj[0].conjugate()*state[0] + params.proj[1].conjugate()*state[1])

    elif spin == 'one': 
        # compute initial probability
        probs[0] = absquare(params.proj[0].conjugate()*state[0] + params.proj[1].conjugate()*state[1] + params.proj[2].conjugate()*state[2])

        # iterate through time using second order expansion
        unitary33(params.dt, B, U)
        for i in range(1,steps):
            # apply half step unitary at current position
            updateState33(state, U)
            #unitary33(time[i], &params, state, state)
 
            t += 2*params.dt
            fastB(t, &params, B)
            unitary33(params.dt, B, U)
            updateState33(state, U)
            if i % params.savef == 0:       
                #store state and store probability along specified projection
                states[i/params.savef][0] = state[0]
                states[i/params.savef][1] = state[1]
                states[i/params.savef][2] = state[2]
                probs[i/params.savef] = absquare(params.proj[0].conjugate()*state[0] + params.proj[1].conjugate()*state[1] + params.proj[2].conjugate()*state[2])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def n_propagateF(paramdict, double complex [:,:] states, double [:,:] F_arr, spin):
    """
    Fast unitary evolution over <time> given simulation parameters <paramdict>.
    """
    import mytime
    # inialise C-type dictionary -> structure
    cdef Params params
    params = paramdict
    cdef int dim
    if spin == 'half':
        dim = 2
    else: #spin == 1
        dim = 3

    # define loop constants
    cdef int j
    cdef unsigned long int i
    cdef unsigned long int steps = <int>ceil((params.tend-params.tstart)/params.dt) + 1

    #Initialise state array
    cdef double complex state[3]

    for j in range(0,dim):
        state[j] = states[0][j]

    #Intialise cos and sin values for fast field calculations
    params.cdtrabi = cos(M_TAU*params.rff*params.dt                 )#2*Pi*Rabi*cos(2*Pi*rff*dt) - constant over simulation
    params.sdtrabi = sin(M_TAU*params.rff*params.dt                 ) #2*Pi*Rabi*sin(2*Pi*rff*dt) - constant over simulation
    params.ctcrabi = cos(M_TAU*params.rff*params.tstart + params.rph) #cos(2*Pi*rff*t) - where t is the 'current simulation' time. This changes over simulation
    params.stcrabi = sin(M_TAU*params.rff*params.tstart + params.rph)  #sin(2*Pi*rff*t) - where t is the 'current simulation' time. This changes over simulation

    params.cdtxl   = cos(M_TAU*params.xlfreq*params.dt                  )#2*Pi*Rabi*cos(2*Pi*rff*dt) - constant over simulation
    params.sdtxl   = sin(M_TAU*params.xlfreq*params.dt                  ) #2*Pi*Rabi*sin(2*Pi*rff*dt) - constant over simulation
    params.ctcxl   = cos(M_TAU*params.xlfreq*params.tr2 + params.xlphase) #cos(2*Pi*rff*t) - where t is the 'current simulation' time. This changes over simulation
    params.stcxl   = sin(M_TAU*params.xlfreq*params.tr2 + params.xlphase)  #sin(2*Pi*rff*t) - where t is the 'current simulation' time. This changes over simulation

    #print(params.xlamp,params.cdtxl,params.sdtxl,params.stcxl)
    # set time step to half
    cdef double t = params.tstart
    params.dt = params.dt*0.5

    cdef double B[4]

    B[0] = M_TAU*params.rabi*params.ctcrabi
    if params.tr2 == params.tstart:
        #print('yay')
        B[0] += M_TAU*params.xlamp*params.ctcxl
    B[1] = 0.0
    B[2] = M_TAU*params.larmor
    B[3] = M_TAU*params.quad

    cdef double complex F[3]
    cdef double complex U[3][3] #unitary operator for spin 1

    if spin == 'half':
        # compute initial probability
        F[0] = 0.5*(state[0]*state[1].conjugate()) #0.5(a.b*+b.a*)
        F[0] += F[0].conjugate()
        F[1] = 0.5j*(state[0]*state[1].conjugate()) #0.5(a.b*+b.a*)
        F[1] += F[1].conjugate()
        F[2] = 0.5*(state[0]*state[0].conjugate()-state[1]*state[1].conjugate())
        F_arr[0][0] = F[0].real
        F_arr[0][1] = F[1].real
        F_arr[0][2] = F[1].real

        # iterate through time using second order expansion
        for i in range(1,steps):
            # apply half step unitary at current position
            unitary22(params.dt, state, B)

            t += 2*params.dt
            fastB(t, &params, B)
            # apply half step at destination time
            unitary22(params.dt, state, B)
            #store state and store probability along specified projection
            if i % params.savef == 0:
                states[i/params.savef][0] = state[0]
                states[i/params.savef][1] = state[1]

                F[0] = 0.5*(state[0]*state[1].conjugate()) #0.5(a.b*+b.a*)
                F[1] = F[0]*1j
                F[0] += F[0].conjugate()
                F[1] = 0.5j*(state[0]*state[1].conjugate()) #0.5(a.b*+b.a*)
                F[1] += F[1].conjugate()
                F[2] = 0.5*(absquare(state[0])-absquare(state[1]))
                
                F_arr[i/params.savef][0] = F[0].real
                F_arr[i/params.savef][1] = F[1].real
                F_arr[i/params.savef][2] = F[2].real

    elif spin == 'one':
        # compute initial probability

        F[0] = invsqrt2*state[1].conjugate()*(state[0]+state[2]) #0.5(a.b*+b.a*)
        F[0] += F[0].conjugate()
        F[1] = invsqrt2*1j*(state[0]-state[2])*state[1].conjugate() #0.5(a.b*+b.a*)
        F[1] += F[1].conjugate()
        F[2] = state[0]*state[0].conjugate()-state[2]*state[2].conjugate()

        F_arr[0][0] = F[0].real
        F_arr[0][1] = F[1].real
        F_arr[0][2] = F[2].real

        # iterate through time using second order expansion
        unitary33(params.dt, B, U)
        for i in range(1,steps):
            # apply half step unitary at current position
            updateState33(state, U)
            #unitary33(time[i], &params, state, state)
 
            t += 2*params.dt
            fastB(t, &params, B)
            unitary33(params.dt, B, U)
            updateState33(state, U)
            #print('state', state[0], state[1], state[2])


            if i % params.savef == 0:
                states[i/params.savef][0] = state[0]
                states[i/params.savef][1] = state[1]
                states[i/params.savef][2] = state[2]

                F[0] = invsqrt2*state[1].conjugate()*(state[0]+state[2]) #0.5(a.b*+b.a*)
                F[0] += F[0].conjugate()
                F[1] = invsqrt2*1j*(state[0]-state[2])*state[1].conjugate() #0.5(a.b*+b.a*)
                F[1] += F[1].conjugate()
                F[2] = state[0]*state[0].conjugate()-state[2]*state[2].conjugate()
                
                F_arr[i/params.savef][0] = F[0].real
                F_arr[i/params.savef][1] = F[1].real
                F_arr[i/params.savef][2] = F[2].real
         
                #print('F',F[0],F[1],F[2])

# computes an updated state given an input and the 3 B field components
@cython.boundscheck(False) # deactivate index bound checks locally
@cython.wraparound(False)  # deactivate index wraparound check
@cython.nonecheck(False)   # deactivate none checks (these will segfault if you pass a None-type)
@cython.cdivision(True)
cdef inline void unitary22(double dt, double complex state[2], double B[3]):
    # define intermediate variables
    cdef double norma,normb, E, c, s, x, y, z
    cdef double complex phase, psi_up, psi_down
    cdef double complex U11, U22, U12, U21, sa, sb
    ####print(np.asarray(B))

    #Multiply by time incremement to get unitary evolver
    x = B[0]*dt
    y = B[1]*dt
    z = B[2]*dt*0.5

    ####print(np.asarray(B))
    
    # can't handle a==b==0 with cdivision activated
    if abs(x) < 1e-100 and abs(y) < 1e-100:
        x = 1e-100
    
    # compute energy
    E = sqrt(x*x+y*y+z*z)

    # compute eigenvalue while still local (definitely don't want trig functions to wait on L3)
    c = cos(E)
    s = sin(E)
    phase = c + 1j*s
    #phase = cos(E) + 1j*sin(E)

    # compute unnormalised eigenvectors (force casting after some strange assignment glitch)
    U11 = <complex>(z - E)
    U12 = <complex>(z + E)
    U21 = (x + 1j * y)
    U22 = U21
    
    # compute unitary by applying decomposition to state piecewise
    psi_down = U11.conjugate()*state[0] + U21.conjugate()*state[1]
    psi_up = U12.conjugate()*state[0] + U22.conjugate()*state[1]
    
    # apply phase factor
    psi_down *= phase
    psi_up *= phase.conjugate()

    # compute squared normalisation
    norma = absquare(U11) + absquare(U21)
    normb = absquare(U12) + absquare(U22)
    
    # apply final eigenvector and save to output state memory location
    state[0] = U11*psi_down/norma + U12*psi_up/normb
    state[1] = U21*psi_down/norma + U22*psi_up/normb


@cython.boundscheck(False) # deactivate index bound checks locally
@cython.wraparound(False)  # deactivate index wraparound check
@cython.nonecheck(False)   # deactivate none checks (these will segfault if you pass a None-type)
@cython.cdivision(True)
cdef void unitary33(double dt, double B[4], double complex U[3][3]):    
 #based on the algorithm dsyevh3
# ----------------------------------------------------------------------------
# Calculates the eigenvalues and normalized eigenvectors of a hermitian 3x3
# matrix A using Cardano's method for the eigenvalues and an analytical
# method based on vector cross products for the eigenvectors. However,
# if conditions are such that a large error in the results is to be
# expected, the routine falls back to using the slower, but more
# accurate QL algorithm. Only the diagonal and upper triangular parts of A need
# to contain meaningful values. Access to A is read-only.
# ----------------------------------------------------------------------------
# Parameters:
#   A: The hermitian input matrix
#   Q: Storage buffer for eigenvectors
#   w: Storage buffer for eigenvalues
# ----------------------------------------------------------------------------
# Return value:
#   0: Success
#  -1: Error
# ----------------------------------------------------------------------------
# Dependencies:
#   zheevc3(), zhetrd3(), QL33()
# ----------------------------------------------------------------------------
    cdef double w[3] #eigenvalues
    cdef double complex Q[3][3] #eigenvectors

    cdef double norm          # Squared norm or inverse norm of current eigenvector
    cdef double error         # Estimated maximum roundoff error
    cdef double t, u          # Intermediate storage
    cdef int j                # Loop counter

  # Calculate eigenvalues
    ####print(B[0],B[1],B[2])
    eval33(B, w)
    ####print(w[0],w[1],w[2])
    #Calculates maximum eigenvalue and sets error bound

    t = abs(w[0])
    u = abs(w[1])
    if (u > t):
        t = u
    u = abs(w[2])
    if (u > t):
        t = u
    if (t < 1.0):
        u = t
    else:
        u = t*t
    error = 256.0 * eps * u*u

    cdef double complex H0, H1
    cdef double B2x, B2y

    B2x = 0.5*B[0]*B[0]
    B2y = 0.5*B[1]*B[1]

    H0 = B[2] + B[3] #try treating as a double or complex double
    H1 = (B[0]-1j*B[1])*invsqrt2

    Q[0][1] = B2x - B2y - 1j*B[0]*B[1]
    Q[1][1] = -H0*H1
    Q[2][1] = B2x+B2y

    # Calculate first eigenvector by the formula
    Q[0][0] = Q[0][1]
    Q[1][0] = Q[1][1] + H1*w[0]
    Q[2][0] = (H0 - w[0]) * (- w[0]) - Q[2][1]
    norm    = absquare(Q[0][0]) + absquare(Q[1][0]) + square(Q[2][0].real)

    # If vectors are nearly linearly dependent, or if there might have
    # been large cancellations in the calculation of A(I,I) - W(1), fall
    # back to QL algorithm
    # Note that this simultaneously ensures that multiple eigenvalues do
    # not cause problems: If W(1) = W(2), then A - W(1) * I has rank 1,
    # i.e. all columns of A - W(1) * I are linearly dependent.
    '''if (norm <= error):
        QL33(B, Q, w)
    else:'''                      # This is the standard branch
    norm = sqrt(1.0 / norm)
    if norm <= 1e-20:
        print('fuck')
    for j in range(0,3):
        Q[j][0] = Q[j][0] * norm
    
    
    # Calculate second eigenvector by the formula
    #   v[1] =  (A - w.conjugate()1]).e1 x (A - w[1]).e2 )
    Q[0][1]  = Q[0][1]
    Q[1][1]  = Q[1][1] + H1*w[1]
    Q[2][1]  = (H0 - w[1]) * (- w[1]) - Q[2][1].real
    #norm     = absquare(Q[0][1]) + absquare(Q[1][1]) + square(Q[2][1].real)
    norm     = (Q[0][1].conjugate()*Q[0][1] + Q[1][1].conjugate()*Q[1][1]).real + (Q[2][1].real)**2

    if (norm <= error):
        QL33(B, Q, w)
    else:
        if norm <= 1e-20:
            print('you')

        norm = sqrt(1.0 / norm)
        for j in range(0,3):
            Q[j][1] = Q[j][1] * norm
    
    # Calculate third eigenvector according to
    #   v[2] = v[0] x .conjugate()[1])
    Q[0][2] = (Q[1][0]*Q[2][1] - Q[2][0]*Q[1][1]).conjugate()
    Q[1][2] = (Q[2][0]*Q[0][1] - Q[0][0]*Q[2][1]).conjugate()
    Q[2][2] = (Q[0][0]*Q[1][1] - Q[1][0]*Q[0][1]).conjugate()

    cdef double complex e[3]

    w[0] *= dt 
    w[1] *= dt 
    w[2] *= dt     

    e[0] = cos(w[0])-1j*sin(w[0])
    e[1] = cos(w[1])-1j*sin(w[1])
    e[2] = cos(w[2])-1j*sin(w[2])

    U[0][0] = e[0]*Q[0][0].conjugate()*Q[0][0] + e[1]*Q[0][1].conjugate()*Q[0][1] + e[2]*Q[0][2].conjugate()*Q[0][2]
    U[1][0] = e[0]*Q[0][0].conjugate()*Q[1][0] + e[1]*Q[0][1].conjugate()*Q[1][1] + e[2]*Q[0][2].conjugate()*Q[1][2]
    U[2][0] = e[0]*Q[0][0].conjugate()*Q[2][0] + e[1]*Q[0][1].conjugate()*Q[2][1] + e[2]*Q[0][2].conjugate()*Q[2][2]
    U[0][1] = e[0]*Q[1][0].conjugate()*Q[0][0] + e[1]*Q[1][1].conjugate()*Q[0][1] + e[2]*Q[1][2].conjugate()*Q[0][2]
    U[1][1] = e[0]*Q[1][0].conjugate()*Q[1][0] + e[1]*Q[1][1].conjugate()*Q[1][1] + e[2]*Q[1][2].conjugate()*Q[1][2]
    U[2][1] = e[0]*Q[1][0].conjugate()*Q[2][0] + e[1]*Q[1][1].conjugate()*Q[2][1] + e[2]*Q[1][2].conjugate()*Q[2][2]
    U[0][2] = e[0]*Q[2][0].conjugate()*Q[0][0] + e[1]*Q[2][1].conjugate()*Q[0][1] + e[2]*Q[2][2].conjugate()*Q[0][2]
    U[1][2] = e[0]*Q[2][0].conjugate()*Q[1][0] + e[1]*Q[2][1].conjugate()*Q[1][1] + e[2]*Q[2][2].conjugate()*Q[1][2]
    U[2][2] = e[0]*Q[2][0].conjugate()*Q[2][0] + e[1]*Q[2][1].conjugate()*Q[2][1] + e[2]*Q[2][2].conjugate()*Q[2][2]

    '''print('B',B[0],B[1],B[2],B[3])
    print('U')
    print(U[0][0],U[0][1],U[0][2])
    print(U[1][0],U[1][1],U[1][2])
    print(U[2][0],U[2][1],U[2][2])'''

#Intitalisation of evolving field paramaters required to use fastB()
#Calculates sines and cosines of 2*Pi*rabi*dt to allow fast evolution 
@cython.boundscheck(False) # deactivate index bound checks locally
@cython.wraparound(False)  # deactivate index wraparound check
@cython.nonecheck(False)   # deactivate none checks (these will segfault if you pass a None-type)
cdef inline void initFastB(Params *params):
    params.cdtrabi = cos(M_TAU*params.rff*params.dt)#2*Pi*Rabi*cos(2*Pi*rff*dt) - constant over simulation
    params.sdtrabi = sin(M_TAU*params.rff*params.dt) #2*Pi*Rabi*sin(2*Pi*rff*dt) - constant over simulation
    params.ctcrabi = 1.0 #cos(2*Pi*rff*t) - where t is the 'current simulation' time. This changes over simulation
    params.stcrabi = 0.0 #sin(2*Pi*rff*t) - where t is the 'current simulation' time. This changes over simulation


# computes the B field given a time step and precalculated cos and sin values.
#to use first initialise with initFastB
#fast evolver using trig expansion
@cython.boundscheck(False) # deactivate index bound checks locally
@cython.wraparound(False)  # deactivate index wraparound check
@cython.nonecheck(False)   # deactivate none checks (these will segfault if you pass a None-type)
cdef inline void fastB(double t, Params *params, double B[4]):
    #B[3] = params.quad #as is constant don't need to update
    cdef double tmp #temporary storage for field calculations
    B[0] = (params.cdtrabi*params.ctcrabi - params.sdtrabi*params.stcrabi) #M_TAU*params.rabi*cos(M_TAU*(params.rff)*t)
    params.stcrabi = (params.cdtrabi*params.stcrabi+params.sdtrabi*params.ctcrabi)
    params.ctcrabi = B[0]
    B[0] *= M_TAU*params.rabi
    #B[1] = 0.0 #as is constant don't need to update
    B[2] = M_TAU*params.larmor

    if params.xlamp != 0.0 and t >= params.tr2:
            #print('k')
            tmp = params.cdtxl*params.ctcxl - params.sdtxl*params.stcxl
            B[0] += M_TAU*params.xlamp*tmp 
            params.stcxl = (params.cdtxl*params.stcxl + params.sdtxl*params.ctcxl)
            params.ctcxl = tmp



# computes the B field given a time step and an input array. 
@cython.boundscheck(False) # deactivate index bound checks locally
@cython.wraparound(False)  # deactivate index wraparound check
@cython.nonecheck(False)   # deactivate none checks (these will segfault if you pass a None-type)
cdef inline void getField(double t, Params *params, double B[4]):
    # define omega, detune, wrap
    cdef double omegaD, detune

    B[3] = M_TAU*params.quad
    B[0] = 0.0
    B[1] = 0.0
    B[2] = M_TAU*params.larmor

    # check if detuning run needs to be applied
    if t < params.dett:
        # define generic fields using structure constants  
        if params.rabi != 0.0:
            B[0] += M_TAU*params.rabi*cos(M_TAU*(params.rff)*t)
        if params.xlamp != 0.0:
            B[0] += params.xlamp*sin(M_TAU*params.xlfreq*t + params.xlphase)#*params.dt
        
        # add neural impulse
        if params.zlamp != 0.0:
           B[2] += params.zlamp*sin(M_TAU*params.zlfreq*t + params.zlphase)#*params.dt
        if t >= params.nt and t <= params.nte:
           B[2] += params.sA*sin(M_TAU*params.nf*(t-params.nt))


    elif t < params.dete and params.beta > 0.0:        
        omegaD = params.rabi*sech(params.beta*(t-params.dett))
        detune = params.detA*tanh(params.beta*(t-params.dett))
        B[0] = M_TAU*(omegaD*cos(M_TAU*(params.rff+detune)*(t-params.dett)))

    else:
        # detuning truncation
        B[0] = params.xlamp*sin(M_TAU*params.xlfreq*t + params.xlphase)#*params.dt
        B[2] += params.zlamp*sin(M_TAU*params.zlfreq*t + params.zlphase)#*params.dt



@cython.boundscheck(False) # deactivate index bound checks locally
@cython.wraparound(False)  # deactivate index wraparound check
@cython.nonecheck(False)   # deactivate none checks (these will segfault if you pass a None-type)
def Bfield(double [:] time, paramdict, double [:,:] Bfield):
    """
    Computes B over the specifed time range
    """
    # inialise C-type dictionary -> structure
    cdef Params params
    params = paramdict
    cdef double B[4]

    #Intialise cos and sin values for fast field calculations
    params.cdtrabi = cos(M_TAU*params.rff*params.dt                 )#2*Pi*Rabi*cos(2*Pi*rff*dt) - constant over simulation
    params.sdtrabi = sin(M_TAU*params.rff*params.dt                 ) #2*Pi*Rabi*sin(2*Pi*rff*dt) - constant over simulation
    params.ctcrabi = cos(M_TAU*params.rff*params.tstart + params.rph) #cos(2*Pi*rff*t) - where t is the 'current simulation' time. This changes over simulation
    params.stcrabi = sin(M_TAU*params.rff*params.tstart + params.rph)  #sin(2*Pi*rff*t) - where t is the 'current simulation' time. This changes over simulation

    params.cdtxl   = cos(M_TAU*params.xlfreq*params.dt                  )#2*Pi*Rabi*cos(2*Pi*rff*dt) - constant over simulation
    params.sdtxl   = sin(M_TAU*params.xlfreq*params.dt                  ) #2*Pi*Rabi*sin(2*Pi*rff*dt) - constant over simulation
    params.ctcxl   = cos(M_TAU*params.xlfreq*params.tr2 + params.xlphase) #cos(2*Pi*rff*t) - where t is the 'current simulation' time. This changes over simulation
    params.stcxl   = sin(M_TAU*params.xlfreq*params.tr2 + params.xlphase)  #sin(2*Pi*rff*t) - where t is the 'current simulation' time. This changes over simulation

    #print(params.xlamp,params.cdtxl,params.sdtxl,params.stcxl)
    # set time step to half
    cdef double t = params.tstart
    params.dt *= 0.5

    B[0] = M_TAU*params.rabi*params.ctcrabi
    if params.tr2 == params.tstart:
        #print('yay')
        B[0] += M_TAU*params.xlamp*params.ctcxl
    B[1] = 0.0
    B[2] = M_TAU*params.larmor
    B[3] = M_TAU*params.quad

    # define loop constants
    cdef int i, j
    cdef unsigned long int steps = <int>ceil((params.tend-params.tstart)/params.dt)
    print('k')
    # iterate through time
    for i in range(0,steps):
        fastB(time[i], &params, B)
        for j in range(0,4):
            Bfield[i][j] = B[j]
    
    print('yay')


@cython.boundscheck(False) # deactivate index bound checks locally
@cython.wraparound(False)  # deactivate index wraparound check
@cython.nonecheck(False)   # deactivate none checks (these will segfault if you pass a None-type)
cdef inline void updateState33(double complex state[3], double complex U[3][3]):
    cdef double complex state_out[3]
    state_out[0] = U[0][0] * state[0] + U[0][1] * state[1] + U[0][2] * state[2] 
    state_out[1] = U[1][0] * state[0] + U[1][1] * state[1] + U[1][2] * state[2]
    state_out[2] = U[2][0] * state[0] + U[2][1] * state[1] + U[2][2] * state[2]
    
    state[0]  = state_out[0]
    state[1]  = state_out[1]
    state[2]  = state_out[2]

    ##print('U')
    ##print(U[0][0], U[0][1], U[0][2])
    ##print(U[1][0], U[1][1], U[1][2])
    ##print(U[2][0], U[2][1], U[2][2])


@cython.boundscheck(False) # deactivate index bound checks locally
@cython.wraparound(False)  # deactivate index wraparound check
@cython.nonecheck(False)   # deactivate none checks (these will segfault if you pass a None-type)
@cython.cdivision(True)
# ----------------------------------------------------------------------------
cdef inline void eval33(double B[4], double w[3]):
    #Based on roots of cubic using trig https://en.wikipedia.org/wiki/Cubic_function
    # ----------------------------------------------------------------------------
    # Calculates the eigenvalues of a hermitian 3x3 matrix A using Cardano's
    # analytical algorithm.
    # Only the diagonal and upper triangular parts of A are accessed. The access
    # is read-only.
    # ----------------------------------------------------------------------------
    # Parameters:
    #   A: The hermitian input matrix
    #   w: Storage buffer for eigenvalues
    
    cdef double r, s, u, c, q2, B2

    B2 = B[0]*B[0]+B[1]*B[1]+B[2]*B[2]

    if B[3] == 0.0:
        #With no quadratic Zeeman shift |B| defines splitting
        w[2] = sqrt(B2)
        w[1] = 0.0
        w[0] = -w[2]       
    else:
        #Use transformation from wiki and solve cubic equation with q
        #From wiki r=Sqrt[-p/3], s=-q/2
        #transform det(H-wI)=0 with w=X+2q/3
        #solve transformed cubic equation. Transform back to get w's
        q2 = B[3]*B[3]   
        r = sqrt((q2/3.0+B2)/3.0)
        s = -B[3]*(q2/27.0+B2/6.0-0.5*B[2]*B[2])
        ####print(r)
        u = acos(s/(r*r*r))/3.0
        #sincos(u, s, c) #computes sin and cos simultaneously
        c = cos(u)
        s = sin(u)

        c *= r
        s *= M_SQRT3*r

        #Transform back to get roots
        w[0] = 2.0*B[3]/3.0
        w[1] = w[0]
        w[2] = w[0]

        w[0] += 2.0*c
        w[1] += s-c
        w[2] += -s-c



@cython.boundscheck(False) # deactivate index bound checks locally
@cython.wraparound(False)  # deactivate index wraparound check
@cython.nonecheck(False)   # deactivate none checks (these will segfault if you pass a None-type)
@cython.cdivision(True)
cdef void QL33(double B[4], double complex Q[3][3], double w[3]):
# ----------------------------------------------------------------------------
# Calculates the eigenvalues and normalized eigenvectors of a hermitian 3x3
# matrix A using the QL algorithm with implicit shifts, preceded by a
# Householder reduction to real tridiagonal form.
# The function accesses only the diagonal and upper triangular parts of A.
# The access is read-only.
# ----------------------------------------------------------------------------
# Parameters:
#   A: The hermitian input matrix
#   Q: Storage buffer for eigenvectors
#   w: Storage buffer for eigenvalues
# ----------------------------------------------------------------------------
# Return value:
#   0: Success
#  -1: Error (no convergence)
# ----------------------------------------------------------------------------
    cdef double e[3]                 # The third element is used only as temporary workspace
    cdef double g, r, p, f, b, s, c  # Intermediate storage
    cdef double complex t
    cdef int nIter
    cdef int m, l, i

    w[0] = B[3] + B[2]
    w[1] = 0.0
    w[2] = B[3] - B[2] 

    #store unitary transform to real tridiagonal form in eigenvectors matrix
    for i in range(0,3):
        Q[i][i] = 1.0
        for l in range(0,i):
            Q[i][l] = 0.0
            Q[l][i] = 0.0 

    # Transform A to real tridiagonal form if By/=0
    #            [ w[0]  e          ]
    #    A = Q . [  e    w[1]     e ] . Q^T
    #            [       e     w[2] ]
    # The function accesses only the dia
    if B[1] == 0:
        e[0] = B[0]*invsqrt2
    else:    
        e[0] = sqrt(B[0]*B[0] + B[1]*B[1])
        Q[0][0] = (B[0]-1j*B[1])/e[0] #Unitary matrix is change of basis x-y fields point along x-axis
        Q[2][2] = Q[0][0].conjugate()
        e[0] *= invsqrt2
    e[1] = e[0]
 
    # Calculate eigensystem of the remaining real symmetric tridiagonal matrix
    # with the QL method
    #
    # Loop over all off-diagonal elements
    for l in range(0,2):
        nIter = 0
        while True:
            # Check for convergence and exit iteration loop if off-diagonal
            # element e(l) is zero
            for m in range(l,2):
                g = abs(w[m])+abs(w[m+1])
                if (abs(e[m]) + g == g):
                    break
            if (m == l):
                break

            if (nIter >= 30):
                print("Error. QL33 doesn't converge")
                exit()
            nIter += 1
            # Calculate g = d_m - k
            g = 0.5*(w[l+1] - w[l]) /e[l]
            r = sqrt(g*g + 1.0)
            if (g > 0):
                g = w[m] - w[l] + e[l]/(g + r)
            else:
                g = w[m] - w[l] + e[l]/(g - r)

            s = 1.0
            c = 1.0
            p = 0.0
            for i in range(m-1, l - 1, -1):
                f = s * e[i]
                b = c * e[i]
                if (abs(f) > abs(g)):
                    c      = g / f
                    r      = sqrt(c*c + 1.0)
                    e[i+1] = f * r
                    s      = 1.0/r
                    c     *= s
                else:
                    s      = f / g
                    r      = sqrt(s*s + 1.0)
                    e[i+1] = g * r
                    c      = 1.0/r
                    s     *= c   
            g = w[i+1] - p
            r = (w[i] - g)*s + 2.0*c*b
            p = s * r
            w[i+1] = g + p
            g = c*r - b

            # Form eigenvectors
            for k in range(0, 3):
                t = Q[k][i+1]
                Q[k][i+1] = s*Q[k][i] + c*t
                Q[k][i]   = c*Q[k][i] - s*t
            
            w[l] -= p
            e[l]  = g
            e[m]  = 0.0