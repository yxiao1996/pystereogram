from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

try:
    with open("readme.md", 'r') as f:
        long_description = f.read()
except:
    long_description = ''

setup(
    name='pystereogram',
    version="0.0.13",
    packages=find_packages(),
    author="Yu Xiao",
    description="A library supporting efficient stereogram image computation.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/yxiao1996/pystereogram",
    ext_modules=cythonize([Extension("compute_line", ["autostereogram/compute_line.pyx"])]),
    include_dirs=[numpy.get_include()],
    install_requires=[
        'numpy>=1.19.2'
    ],
    zip_safe=False,
    package_data={"autostereogram": ["compute_line.pyx"]}
)