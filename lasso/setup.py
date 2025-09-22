from pathlib import Path
from setuptools import setup

def read(fname):
    return (Path(__file__).parent / fname).open().read()

setup(
    name='lasso', 
    version='0.0.1', 
    description='Implementation of Lasso (Tibshirani 1996) from scratch', 
    long_description=read('README.md'), 
    url="https://github.com/lleomllr/lasso",
    packages=["lasso"],
    install_requires=[
        'numpy', 
        'pandas', 
        'scikit-learn', 
        'matplotlib'
    ], 
    python_requires='>=3.8'
)