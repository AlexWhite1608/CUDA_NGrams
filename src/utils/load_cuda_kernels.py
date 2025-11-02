import os
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Load CUDA kernel code from file
def load_cuda_kernels(kernel_file: str = None):

    if kernel_file and os.path.exists(kernel_file):
        with open(kernel_file, 'r') as f:
            kernel_code = f.read()
    
    # compiles and returns the CUDA module
    mod = SourceModule(kernel_code)
    return mod
