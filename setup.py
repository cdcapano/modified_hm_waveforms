#!/usr/bin/env python
"""
setup.py file for modified higher mode waveforms plugin package
"""

from setuptools import Extension, setup, Command
from setuptools import find_packages

VERSION = '0.0.dev0'

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='modified_hm_waveforms',
    version=VERSION,
    description='Waveform hook that allows different parameters to be used '
                'for different modes.',
    long_description=long_description,
    author='Collin Capano',
    author_email='cdcapano@gmail.com',
    keywords=['pycbc', 'signal processing', 'gravitational waves'],
    install_requires=['pycbc', 'lalsuite'],
    py_modules=['modhm'],
    entry_points={
        "pycbc.waveform.fd" : " = modhm:modhm_fd"},
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ],
)
