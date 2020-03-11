import glob
import importlib
import os
import re
from setuptools import setup, find_packages

PACKAGE_NAME = "dnntime"

here = os.path.abspath(os.path.dirname(__file__))
info = {}
with open(os.path.join(here, PACKAGE_NAME, '__version__.py'), 'r') as f:
    exec(f.read(), info)
with open("README.md", "r") as fh:
    READ_ME = fh.read()


setup(
    name=info['__title__'],
    author=info['__author__'],
    description=info['__description__'],
    url=info['__url__'],
    version=info['__version__'],
    license=info['__license__'],
    long_description=READ_ME,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests",)),
    python_requires='>= 3.6',
    install_requires=[
        'pyyaml >= 5.3',
        'numpy >= 1.18',
        'pandas == 0.25.3',
        'matplotlib >= 3.1.3',
        'seaborn >= 0.9.0',
        'plotly >= 4.5.3',
        'nbformat >= 5.0.4',
        'scikit-learn >= 0.22.0',
        'statsmodels >= 0.11.0',
        'tscv >= 0.0.4',
        'art >= 4.5',
        'fbprophet >= 0.5',
        'tensorflow >= 2.1.0',
        'ipdb >= 0.13.0',
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "Operating System :: OS Independent",
    ],
)
