from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import os
import numpy as np

#compile with 'python setup.py build_ext --inplace'

# use this for compiling with MSVC
# ext_modules = [
#     Extension(
#         "quantumc",
#         ["quantumc.pyx"],
#         include_dirs=[np.get_include()],
#         extra_compile_args=["/openmp"],
#         extra_link_args=["/openmp"]
#     )
# ]

# use this for compiling with gcc
ext_modules = [
    Extension(
        "quantumc",
        ["quantumc.pyx"],
        include_dirs=[np.get_include()]
        ,extra_compile_args=["-O3"]
    )
]

setup(name='quantumc',
      ext_modules=cythonize(ext_modules, annotate=True,)
)