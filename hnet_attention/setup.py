from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0")  # A100 default

setup(
    name="hnet_attention",
    ext_modules=[
        CUDAExtension(
            name="hnet_attention_cuda",
            sources=[
                "attention_extension.cpp",
                "attention_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-Xptxas=-v",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
