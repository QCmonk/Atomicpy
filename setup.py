from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import os
import numpy as np

from autocython import PLATFORM_SUFFIX

print(PLATFORM_SUFFIX)
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
        "quantumc"+PLATFORM_SUFFIX,
        ["quantumc.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O2" ,"-ffast-math","-funsafe-math-optimizations"]
    )
]

setup(name='quantumc',
      ext_modules=cythonize(ext_modules, annotate=True,)
)

