try:
    from . import cuda_kernels
    swiglu_fwd_cuda = cuda_kernels.swiglu_fwd
    swiglu_bwd_cuda = cuda_kernels.swiglu_bwd
except ImportError:
    swiglu_fwd_cuda = None
    swiglu_bwd_cuda = None

