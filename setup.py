from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='Umu_blobs',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='Umu_blobs.cuda_extension',
            sources=[
                'Umu_blobs/cuda/Umu_blobs.cu'
            ],
        )
    ],
    extra_compile_args={
        'nvcc': [
            '-Xcompiler', '-fPIC', '--threads', '16'
        ]
    },
    cmdclass={
        'build_ext': BuildExtension
    }
)
