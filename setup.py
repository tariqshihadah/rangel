from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.7'
DESCRIPTION = 'Range data manipulations and operations'
LONG_DESCRIPTION = """Module featuring RangeCollection object class for the management of range data and optimized performance of various simple and complex range operations, including range and point overlays, equitable and score-based separation of overlapping ranges, generation of random range data, and range data comparisons among others."""

# Setting up
setup(
    name="rangel",
    version=VERSION,
    author="Tariq Shihadah",
    author_email="<tariq.shihadah@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'scipy'],
    keywords=['python', 'range', 'numeric', 'interval'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)