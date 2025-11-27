#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template<typename T>
__global__ void swiglu_fwd_kernel(
    const T* __restrict__ X,
    const T* __restrict__ Y,
    T* __restrict__ OUT,
    int stride_x_row, int stride_y_row, int stride_out_row,
    int ncols, int BLOCK_N
) {
    int row = blockIdx.x;
    int start_col = blockIdx.y * BLOCK_N;
    int col = start_col + threadIdx.x;
    
    if (col < ncols) {
        float x = float(X[row * stride_x_row + col]);
        float y = float(Y[row * stride_y_row + col]);
        float x_sigmoid = 1.0f / (1.0f + expf(-x));
        OUT[row * stride_out_row + col] = T(x * x_sigmoid * y);
    }
}

template<typename T>
__global__ void swiglu_bwd_kernel(
    const T* __restrict__ X,
    const T* __restrict__ Y,
    const T* __restrict__ DOUT,
    T* __restrict__ OUT,
    T* __restrict__ DX,
    T* __restrict__ DY,
    int stride_x_row, int stride_y_row, int stride_dout_row,
    int stride_out_row, int stride_dx_row, int stride_dy_row,
    int ncols, int BLOCK_N, bool recompute_output
) {
    int row = blockIdx.x;
    int start_col = blockIdx.y * BLOCK_N;
    int col = start_col + threadIdx.x;
    
    if (col < ncols) {
        float x = float(X[row * stride_x_row + col]);
        float y = float(Y[row * stride_y_row + col]);
        float dout = float(DOUT[row * stride_dout_row + col]);
        
        float x_sigmoid = 1.0f / (1.0f + expf(-x));
        float dx = x_sigmoid * (1.0f + x * (1.0f - x_sigmoid)) * y * dout;
        float dy = x * x_sigmoid * dout;
        
        DX[row * stride_dx_row + col] = T(dx);
        DY[row * stride_dy_row + col] = T(dy);
        
        if (recompute_output) {
            OUT[row * stride_out_row + col] = T(x * x_sigmoid * y);
        }
    }
}

torch::Tensor swiglu_fwd(torch::Tensor x, torch::Tensor y) {
    auto out = torch::empty_like(x);
    int M = x.size(0);
    int N = x.size(1);
    int BLOCK_N = 256;
    
    dim3 grid(M, (N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(BLOCK_N);
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "swiglu_fwd", [&] {
        swiglu_fwd_kernel<scalar_t><<<grid, block>>>(
            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
            x.stride(0), y.stride(0), out.stride(0), N, BLOCK_N
        );
    });
    
    return out;
}

std::vector<torch::Tensor> swiglu_bwd(
    torch::Tensor x, torch::Tensor y, torch::Tensor dout,
    bool recompute_output
) {
    auto dx = torch::empty_like(x);
    auto dy = torch::empty_like(y);
    torch::Tensor out;
    if (recompute_output) {
        out = torch::empty_like(x);
    }
    
    int M = x.size(0);
    int N = x.size(1);
    int BLOCK_N = 256;
    
    dim3 grid(M, (N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(BLOCK_N);
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "swiglu_bwd", [&] {
        swiglu_bwd_kernel<scalar_t><<<grid, block>>>(
            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), dout.data_ptr<scalar_t>(),
            recompute_output ? out.data_ptr<scalar_t>() : nullptr,
            dx.data_ptr<scalar_t>(), dy.data_ptr<scalar_t>(),
            x.stride(0), y.stride(0), dout.stride(0),
            recompute_output ? out.stride(0) : 0,
            dx.stride(0), dy.stride(0),
            N, BLOCK_N, recompute_output
        );
    });
    
    if (recompute_output) {
        return {dx, dy, out};
    }
    return {dx, dy};
}

