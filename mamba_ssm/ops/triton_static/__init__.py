try:
    from . import cuda_kernels
    swiglu_fwd_cuda = cuda_kernels.swiglu_fwd
    swiglu_bwd_cuda = cuda_kernels.swiglu_bwd
    chunk_scan_fwd_cuda = cuda_kernels.chunk_scan_fwd
except ImportError:
    swiglu_fwd_cuda = None
    swiglu_bwd_cuda = None
    chunk_scan_fwd_cuda = None

