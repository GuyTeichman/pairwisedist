#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read().split('\n')

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
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    description='Calculate the pairwise-distance matrix for an array of *n* samples by *p* features, '
                'sing a selection of distance metrics.',
    install_requires=requirements,
    python_requires='>=3.7',
    license="License :: OSI Approved :: Apache Software License",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords=['distance', 'pairwise distance', 'YS1', 'YR1', 'pairwise-distance matrix', 'Son and Baek dissimilarities',
              'Son and Baek'],
    name='pairwisedist',
    packages=find_packages(include=['pairwisedist']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/GuyTeichman/pairwisedist',
    version='1.3.0',
    zip_safe=False,
)
