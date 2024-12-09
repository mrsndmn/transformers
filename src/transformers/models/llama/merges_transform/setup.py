from setuptools import setup, Extension
from torch.utils import cpp_extension

from setuptools import find_packages, setup


debug_mode = False

extra_compile_args = {
    "cxx": [
        "-O3" if not debug_mode else "-O0",
        "-fdiagnostics-color=always",
    ],
    "nvcc": [
        "-O3" if not debug_mode else "-O0",
    ],
}

setup(
    name='generate_merges',
    packages=find_packages(),
    ext_modules=[cpp_extension.CUDAExtension(
        name='generate_merges._C',
        sources=['generate_merges/csrc/generate_merges.cu'],
        extra_compile_args=extra_compile_args,
    )],
    install_requires=["torch"],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
