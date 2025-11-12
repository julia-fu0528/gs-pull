# python setup.py build_ext --inplace
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

includes_numpy = '-I ' + np.get_include() + ' '
os.environ['CFLAGS'] = includes_numpy + (os.environ['CFLAGS'] if 'CFLAGS' in os.environ else '')
os.environ['CXXFLAGS'] = includes_numpy + (os.environ['CXXFLAGS'] if 'CXXFLAGS' in os.environ else '')

ext_modules = [
    Extension(
        "_marching_cubes_lewiner_cy",
        ["_marching_cubes_lewiner_cy.pyx"],
        include_dirs=[np.get_include()],
        language="c++"
    )
]

setup(
    name="My MC",
    ext_modules=cythonize(ext_modules),
)
