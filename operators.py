import numpy as np
import matplotlib.pyplot as plt


# Plancks constant
pbar = 6.626070040e-34
# reduced
hbar = pbar/(2*np.pi)
# Bohr magneton in J/Gauss
mub = (9.274009994e-24)/1e4
# g factor
gm = 2.00231930436
# Gyromagnetic ratio
gyro = 699.9e3
# pi is pi
pi = np.pi
#sqrt(2)
sqrt2 = np.sqrt(2)
#1/sqrt(2)
invsqrt2 = 0.70710678118654752440084436210484903928483593768847

# identity matrix
_ID = np.asarray([[1, 0], [0, 1]])
# X gate
_X = 0.5*np.asarray([[0, 1], [1, 0]])
# Z gate
_Z = 0.5*np.asarray([[1, 0], [0, -1]])
# Hadamard gate
_H = invsqrt2*np.asarray([[1, 1], [1, -1]])
# Y Gate
_Y = 0.5*np.asarray([[0, -1j], [1j, 0]])
# S gate
_S = np.asarray([[1, 0], [0, 1j]])
# Sdg gate
_Sdg = np.asarray([[1, 0], [0, -1j]])
# T gate
_T = np.asarray([[1, 0], [0, (1 + 1j)*invsqrt2]])
# Tdg gate
_Tdg = np.asarray([[1, 0], [0, (1 - 1j)*invsqrt2]])
# CNOT gate
_CX = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
# zero state
_pz = np.asarray([1,0], dtype=np.complex128)
# one state
_po = np.asarray([0, 1], dtype=np.complex128)
# Pauli Z for spin 1
_Z1 = np.asarray([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
# Pauli X for spin 1
_X1 = invsqrt2*np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
# Pauli Y for spin 1
_Y1 = 1j*invsqrt2*np.asarray([[0, -1, 0], [1, 0, -1], [0, 1, 0]])
# zero state spin 1
_pz1 = np.asarray([1,0,0], dtype=np.complex128)
# one state
_po1 = np.asarray([0,1,0], dtype=np.complex128)
# two state
_pt1 = np.asarray([0,0,1], dtype=np.complex128)
# identity matrix
_ID1 = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#total S^sq operator
_sSq = _ID*3/4
_sSq1 = _ID1*2

# define operators for spin 1/2 
op1 = {'h':   _H,
        'id':  _ID,
        'x':   _X,
        'y':   _Y,
        'z':   _Z,
        't':   _T,
        'tdg': _Tdg,
        's':   _S,
        'sdg': _Sdg,
        'cx':  _CX,
        'pz': _pz,
        'po': _po}




# define operators for spin 1 
op2 = {'x': _X1,
        'y': _Y1,
        'z': _Z1,
        'id': _ID1,
        'pz': _pz1,
        'po': _po1,
        'pt': _pt1,
        'ssq': _sSq1}


# measurement projections for spin 1/2
meas1 = {"0":np.asarray([1,0], dtype=np.complex128),
		 "1":np.asarray([0,1], dtype=np.complex128),
		 "+":np.asarray([1,1]/np.sqrt(2), dtype=np.complex128),
		 "-":np.asarray([1,-1]/np.sqrt(2), dtype=np.complex128),
		 "+i":np.asarray([1,1j]/np.sqrt(2), dtype=np.complex128),
		 "-i":np.asarray([1,-1j]/np.sqrt(2), dtype=np.complex128)
		}

# measurement projections for spin 1
meas2 = {"0":np.asarray([1,0,0], dtype=np.complex128),
                 "1":np.asarray([0,1,0], dtype=np.complex128),
                 "2":np.asarray([0,0,1], dtype=np.complex128),
                 "+":np.asarray([1,1,1]/np.sqrt(3), dtype=np.complex128),
                 "+01":np.asarray([1,1,0]/np.sqrt(2), dtype=np.complex128),
                 "+02":np.asarray([1,0,1]/np.sqrt(2), dtype=np.complex128),
                 "+12":np.asarray([0,1,1]/np.sqrt(2), dtype=np.complex128),   
                 "-0":np.asarray([-1,1,1]/np.sqrt(3), dtype=np.complex128),   
                 "-1":np.asarray([1,-1,1]/np.sqrt(3), dtype=np.complex128),     
                 "-2":np.asarray([1,1,-1]/np.sqrt(3), dtype=np.complex128),                     
                 "-01":np.asarray([1,-1,0]/np.sqrt(2), dtype=np.complex128),
                 "-02":np.asarray([1,0,-1]/np.sqrt(2), dtype=np.complex128),
                 "-12":np.asarray([0,1,-1]/np.sqrt(2), dtype=np.complex128),
                 "m":np.asarray([0.5,invsqrt2,0.5], dtype=np.complex128)}