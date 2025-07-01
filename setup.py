# setup.py
from setuptools import setup, find_packages

setup(
    name='rlhf_icu',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'gymnasium_robotics',
        'minari',
        'torch',
        'numpy',
        'pandas',
        'matplotlib',
        'jax',
        'flax',
        'stable-baselines3',
    ]
)
