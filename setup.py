import os

import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='coralml',
    long_description=long_description,
    packages=setuptools.find_packages(include=["clefcoral*"]),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=[
    ]
)
