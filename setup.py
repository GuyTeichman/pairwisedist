#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy', 'scipy']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Guy Teichman",
    author_email='guyteichman@gmail.com',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    description='',
    install_requires=requirements,
    python_requires='>3.6',
    license="License :: OSI Approved :: Apache Software License",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords=['distance', 'pairwise distance', 'YS1', 'YR1', 'pairwise-distance matrix', 'Son and Baek dissimilarities',
              'Son and Baek'],
    name='pairwisedist',
    packages=find_packages(),
    # packages=find_packages(include=['pairwisedist']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/GuyTeichman/pairwisedist',
    version='1.1.0',
    zip_safe=False,
)
