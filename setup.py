from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy

with open("readme.md", 'r') as f:
    long_description = f.read()

setup(
    name='pystereogram',
    version="0.0.1",
    packages=find_packages(),
    author="Yu Xiao",
    description="A library supporting efficient stereogram image computation.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    ext_modules=cythonize("compute_line.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)