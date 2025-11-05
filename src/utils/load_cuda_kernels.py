import os
from pycuda.compiler import SourceModule

def load_cuda_kernels(kernel_file: str = None):
    if kernel_file and os.path.exists(kernel_file):
        print(f"Loading CUDA kernel from file: {kernel_file}")
        with open(kernel_file, 'r') as f:
            kernel_code = f.read()
        mod = SourceModule(kernel_code)
        return mod
    else:
        raise FileNotFoundError(f"CUDA kernel file not found: {kernel_file}")