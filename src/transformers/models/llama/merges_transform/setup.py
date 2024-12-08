from setuptools import setup, Extension
from torch.utils import cpp_extension

from setuptools import find_packages, setup

setup(
    name='generate_merges',
    packages=find_packages(),
    ext_modules=[cpp_extension.CUDAExtension(
        name='generate_merges._C',
        sources=['generate_merges/csrc/generate_merges.cu'],
    )],
    install_requires=["torch"],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
