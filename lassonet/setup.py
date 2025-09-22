from pathlib import Path
from setuptools import setup

def read(fname):
    return (Path(__file__).parent / fname).open().read()

setup(
    name='lassonet', 
    version='0.0.1', 
    description='Implementation of LassoNet', 
    long_description=read('README.md'), 
    url="https://github.com/lleomllr/lasso/lassonet",
    packages=["lassonet"],
    install_requires=[
        'torch >= 1.11'
        'numpy', 
        'pandas', 
        'scikit-learn', 
        'matplotlib'
    ], 
    python_requires='>=3.8'
)