#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cassert>
#include <tuple>
#include <iostream>


__host__ __device__ inline int div_up(int a, int b) { return (a + b - 1) / b; }

#ifndef BLOCK_M
#define BLOCK_M 32 // rows in chunk (tile of chunk_size)
#endif
#ifndef BLOCK_N
#define BLOCK_N 64 // headdim tile
#endif
#ifndef BLOCK_K
#define BLOCK_K 32 // inner chunk K tile
#endif
#ifndef BLOCK_D
#define BLOCK_D 64 // tile for dstate gemm
#endif

template<typename scalar_t, typename acc_t>
__global__ void chunk_scan_fwd_kernel(
    const scalar_t* __restrict__ cb,            // cb: (batch, nchunks, ngroups, chunk_size, chunk_size)
    const scalar_t* __restrict__ x,             // x: (batch, seqlen, nheads, headdim)
    const scalar_t* __restrict__ z,             // optional z same shape as x
    const scalar_t* __restrict__ dt,            // dt: (batch, nheads, nchunks, chunk_size)
    const scalar_t* __restrict__ dA_cumsum,     // dA_cumsum: (batch, nheads, nchunks, chunk_size)
    const int64_t* __restrict__ seq_idx,        // optional seq_idx: (batch, seqlen) or nullptr
    const scalar_t* __restrict__ C,             // C: (batch, seqlen, ngroups, dstate)
    const scalar_t* __restrict__ states,        // states: (batch, nchunks, nheads, headdim, dstate)
    const scalar_t* __restrict__ D,             // optional D shape either (nheads, headdim) or (nheads,)
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ out_x,
    int batch, int seqlen, int nheads, int headdim,
    int nchunks, int chunk_size, int ngroups, int dstate,
    int64_t stride_cb_batch, int64_t stride_cb_chunk, int64_t stride_cb_group, int64_t stride_cb_m, int64_t stride_cb_k,
    int64_t stride_x_batch, int64_t stride_x_seqlen, int64_t stride_x_head, int64_t stride_x_hdim,
    int64_t stride_dt_batch, int64_t stride_dt_head, int64_t stride_dt_chunk, int64_t stride_dt_csize,
    int64_t stride_dA_batch, int64_t stride_dA_head, int64_t stride_dA_chunk, int64_t stride_dA_csize,
    int64_t stride_C_batch, int64_t stride_C_seqlen, int64_t stride_C_group, int64_t stride_C_dstate,
    int64_t stride_states_batch, int64_t stride_states_chunk, int64_t stride_states_head, int64_t stride_states_hdim, int64_t stride_states_dstate,
    int64_t stride_out_batch, int64_t stride_out_seqlen, int64_t stride_out_head, int64_t stride_out_hdim,
    bool HAS_Z, bool HAS_D, bool D_HAS_HDIM, bool HAS_SEQ_IDX
) {

    const int num_pid_n = div_up(headdim, BLOCK_N);

    int pid_x = blockIdx.x;
    int pid_m = pid_x / num_pid_n;
    int pid_n = pid_x % num_pid_n;

    int pid_bc = blockIdx.y; 
    int pid_b = pid_bc / nchunks;
    int pid_c = pid_bc % nchunks;

    int pid_h = blockIdx.z;

    if (pid_b >= batch || pid_c >= nchunks || pid_h >= nheads) return;

    int heads_per_group = nheads / ngroups;
    int group_idx = pid_h / heads_per_group;

    int m_start = pid_m * BLOCK_M;
    int n_start = pid_n * BLOCK_N;
    int m_len = min(BLOCK_M, chunk_size - m_start);
    int n_len = min(BLOCK_N, headdim - n_start);

    if (m_len <= 0 || n_len <= 0) return;

    const scalar_t* cb_base = cb + pid_b * stride_cb_batch + pid_c * stride_cb_chunk + group_idx * stride_cb_group;
    int seq_start = pid_c * chunk_size;
    const scalar_t* x_base = x + pid_b * stride_x_batch + seq_start * stride_x_seqlen + pid_h * stride_x_head;
    const scalar_t* dt_base = dt + pid_b * stride_dt_batch + pid_h * stride_dt_head + pid_c * stride_dt_chunk;
    const scalar_t* dA_base = dA_cumsum + pid_b * stride_dA_batch + pid_h * stride_dA_head + pid_c * stride_dA_chunk;
    const scalar_t* C_base = C + pid_b * stride_C_batch + (seq_start) * stride_C_seqlen + group_idx * stride_C_group;
    const scalar_t* states_base = states + pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head;

    const scalar_t* z_base = nullptr;
    if (HAS_Z) {
        z_base = z + pid_b * stride_x_batch + seq_start * stride_x_seqlen + pid_h * stride_x_head;
    }
    const int64_t* seq_idx_base = nullptr;
    if (HAS_SEQ_IDX) {
        seq_idx_base = seq_idx + pid_b * seqlen;
    }

    scalar_t* out_base = out + pid_b * stride_out_batch + seq_start * stride_out_seqlen + pid_h * stride_out_head;
    scalar_t* out_x_base = nullptr;
    if (HAS_Z) out_x_base = out_x + pid_b * stride_out_batch + seq_start * stride_out_seqlen + pid_h * stride_out_head;

    acc_t acc[BLOCK_M][BLOCK_N/1];
    for (int i = 0; i < m_len; ++i) {
        for (int j = 0; j < n_len; ++j) acc[i][j] = (acc_t)0;
    }

    // compute contribution from C @ prev_states
    for (int kd = 0; kd < dstate; kd += BLOCK_D) {
        int kd_len = min(BLOCK_D, dstate - kd);

        extern __shared__ char smem_arr[]; 
       
        acc_t C_tile_reg[BLOCK_M][BLOCK_D]; 

        for (int i = 0; i < m_len; ++i) {
            int m_idx = m_start + i;
            scalar_t* C_row = (scalar_t*) (C_base + m_idx * stride_C_seqlen + kd * stride_C_dstate);
            for (int kk = 0; kk < kd_len; ++kk) {
                scalar_t v = __ldg(C_row + kk * stride_C_dstate); // read-only cache
                C_tile_reg[i][kk] = (acc_t)v;
            }
        }

        // Load prev_states tile
        acc_t P_tile_reg[BLOCK_D][BLOCK_N];
        for (int kk = 0; kk < kd_len; ++kk) {
            int k_idx = kd + kk;
            for (int j = 0; j < n_len; ++j) {
                int n_idx = n_start + j;
                // prev_states element at (n_idx, k_idx):
                const scalar_t* pptr = states_base + n_idx * stride_states_hdim + k_idx * stride_states_dstate;
                scalar_t pv = __ldg(pptr);
                P_tile_reg[kk][j] = (acc_t)pv;
            }
        }

        // Multiply: acc += C_tile_reg * P_tile_reg
        for (int i = 0; i < m_len; ++i) {
            for (int kk = 0; kk < kd_len; ++kk) {
                acc_t a = C_tile_reg[i][kk];
                for (int j = 0; j < n_len; ++j) {
                    acc[i][j] += a * P_tile_reg[kk][j];
                }
            }
        }
    }

    // Apply scale from dA_cumsum per m
    acc_t scale_m_arr[BLOCK_M];
    for (int i = 0; i < m_len; ++i) {
        int m_idx = m_start + i;
        const scalar_t* dAptr = dA_base + m_idx * stride_dA_csize;
        acc_t dA = (acc_t) __ldg(dAptr);
        scale_m_arr[i] = expf((float)dA);
    }
    for (int i = 0; i < m_len; ++i) {
        acc_t s = scale_m_arr[i];
        for (int j = 0; j < n_len; ++j) acc[i][j] *= s;
    }

    // accumlate cb * x over k across chunk dimensions
    for (int k0 = 0; k0 < chunk_size; k0 += BLOCK_K) {
        int klen = min(BLOCK_K, chunk_size - k0);

        // load cb_tile into shared memory or registers
        acc_t cb_tile_reg[BLOCK_M][BLOCK_K];
        acc_t x_tile_reg[BLOCK_K][BLOCK_N];

        // preload dt_k and compute combined cb *= dt_k
        acc_t dt_vec[BLOCK_K];
        for (int kk = 0; kk < klen; ++kk) {
            int k_idx = k0 + kk;
            const scalar_t* dtptr = dt_base + k_idx * stride_dt_csize;
        }

        for (int i = 0; i < m_len; ++i) {
            int m_idx = m_start + i;
            for (int kk = 0; kk < klen; ++kk) {
                int k_idx = k0 + kk;
                // compute mask for causal: require m_idx >= k_idx (within chunk offset)
                bool pass = (m_idx >= k_idx); 
                if (!pass) { cb_tile_reg[i][kk] = (acc_t)0; continue; }
                const scalar_t* cbptr = cb_base + m_idx * stride_cb_m + k_idx * stride_cb_k;
                scalar_t v = __ldg(cbptr);
                // multiply by dt scalar
                cb_tile_reg[i][kk] = (acc_t)v * dt_vec[kk];
            }
        }

        // Load x tile
        for (int kk = 0; kk < klen; ++kk) {
            int k_idx = k0 + kk;
            for (int j = 0; j < n_len; ++j) {
                int n_idx = n_start + j;
                const scalar_t* xptr = x_base + k_idx * stride_x_seqlen + n_idx * stride_x_hdim;
                scalar_t xv = __ldg(xptr);
                x_tile_reg[kk][j] = (acc_t)xv;
            }
        }

        // acc += cb_tile_reg * x_tile_reg
        for (int i = 0; i < m_len; ++i) {
            for (int kk = 0; kk < klen; ++kk) {
                acc_t a = cb_tile_reg[i][kk];
                for (int j = 0; j < n_len; ++j) {
                    acc[i][j] += a * x_tile_reg[kk][j];
                }
            }
        }
    }

    // HAS_D from Triton kernel
    if (HAS_D) {
        for (int i = 0; i < m_len; ++i) {
            int m_idx = m_start + i;
            for (int j = 0; j < n_len; ++j) {
                int n_idx = n_start + j;
                acc_t x_res = (acc_t) __ldg(x_base + m_idx * stride_x_seqlen + n_idx * stride_x_hdim);
                acc_t Dv;
                if (D_HAS_HDIM) {
                    Dv = (acc_t) __ldg(D + pid_h * headdim + n_idx);
                } else {
                    Dv = (acc_t) __ldg(D + pid_h);
                }
                acc[i][j] += x_res * Dv;
            }
        }
    }

    // HAS_Z from Triton kernel
    for (int i = 0; i < m_len; ++i) {
        int m_idx = m_start + i;
        for (int j = 0; j < n_len; ++j) {
            int n_idx = n_start + j;
            acc_t val = acc[i][j];
            if (HAS_Z) {
                out_x_base[m_idx * stride_out_seqlen + n_idx * stride_out_hdim] = (scalar_t) val;
                scalar_t zval = __ldg(z_base + m_idx * stride_x_seqlen + n_idx * stride_x_hdim);
                float zf = (float)zval;
                float gated = zf * (1.0f / (1.0f + expf(-zf)));
                val = val * (acc_t)gated;
            }
            out_base[m_idx * stride_out_seqlen + n_idx * stride_out_hdim] = (scalar_t) val;
        }
    }
}

// Host wrapper
std::vector<torch::Tensor> chunk_scan_fwd_cuda_launcher(
    torch::Tensor cb, torch::Tensor x, torch::Tensor z,
    torch::Tensor dt, torch::Tensor dA_cumsum,
    torch::Tensor seq_idx,
    torch::Tensor C, torch::Tensor states, torch::Tensor D,
    bool IS_CAUSAL
) {
    bool HAS_Z = z.defined();
    bool HAS_D = D.defined();
    bool HAS_SEQ_IDX = seq_idx.defined();

    auto batch = (int) cb.size(0);
    auto nchunks = (int) cb.size(1);
    auto ngroups = (int) cb.size(2);
    auto chunk_size = (int) cb.size(3);
    auto seqlen = (int) x.size(1);
    auto nheads = (int) x.size(2);
    auto headdim = (int) x.size(3);
    auto dstate = (int) C.size(3); 

    auto out = torch::empty_like(x);
    torch::Tensor out_x = torch::Tensor();
    if (HAS_Z) out_x = torch::empty_like(x);

    auto cb_strides = cb.strides();
    auto x_strides = x.strides();
    auto dt_strides = dt.strides();
    auto dA_strides = dA_cumsum.strides();
    auto C_strides = C.strides();
    auto states_strides = states.strides();
    auto out_strides = out.strides();

    int64_t stride_cb_batch = cb_strides[0];
    int64_t stride_cb_chunk = cb_strides[1];
    int64_t stride_cb_group = cb_strides[2];
    int64_t stride_cb_m = cb_strides[3];
    int64_t stride_cb_k = cb_strides[4];

    int64_t stride_x_batch = x_strides[0];
    int64_t stride_x_seqlen = x_strides[1];
    int64_t stride_x_head = x_strides[2];
    int64_t stride_x_hdim = x_strides[3];

    int64_t stride_dt_batch = dt_strides[0];
    int64_t stride_dt_head = dt_strides[1];
    int64_t stride_dt_chunk = dt_strides[2];
    int64_t stride_dt_csize = dt_strides[3];

    int64_t stride_dA_batch = dA_strides[0];
    int64_t stride_dA_head = dA_strides[1];
    int64_t stride_dA_chunk = dA_strides[2];
    int64_t stride_dA_csize = dA_strides[3];

    int64_t stride_C_batch = C_strides[0];
    int64_t stride_C_seqlen = C_strides[1];
    int64_t stride_C_group = C_strides[2];
    int64_t stride_C_dstate = C_strides[3];

    int64_t stride_states_batch = states_strides[0];
    int64_t stride_states_chunk = states_strides[1];
    int64_t stride_states_head = states_strides[2];
    int64_t stride_states_hdim = states_strides[3];
    int64_t stride_states_dstate = states_strides[4];

    int64_t stride_out_batch = out_strides[0];
    int64_t stride_out_seqlen = out_strides[1];
    int64_t stride_out_head = out_strides[2];
    int64_t stride_out_hdim = out_strides[3];

    int num_pid_m = div_up(chunk_size, BLOCK_M);
    int num_pid_n = div_up(headdim, BLOCK_N);
    int grid_x = num_pid_m * num_pid_n;
    int grid_y = batch * nchunks;
    int grid_z = nheads;

    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(1); // TODO: we are not using threads inside block (we unrolled everything). Could be optimized to use thread parallelism...

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "chunk_scan_fwd_cuda", ([&] {
        using scalar_t = scalar_t;
        using acc_t = float;
        const scalar_t* cb_ptr = cb.data_ptr<scalar_t>();
        const scalar_t* x_ptr = x.data_ptr<scalar_t>();
        const scalar_t* z_ptr = HAS_Z ? z.data_ptr<scalar_t>() : nullptr;
        const scalar_t* dt_ptr = dt.data_ptr<scalar_t>();
        const scalar_t* dA_ptr = dA_cumsum.data_ptr<scalar_t>();
        const int64_t* seq_ptr = HAS_SEQ_IDX ? seq_idx.data_ptr<int64_t>() : nullptr;
        const scalar_t* C_ptr = C.data_ptr<scalar_t>();
        const scalar_t* states_ptr = states.data_ptr<scalar_t>();
        const scalar_t* D_ptr = HAS_D ? D.data_ptr<scalar_t>() : nullptr;
        scalar_t* out_ptr = out.data_ptr<scalar_t>();
        scalar_t* out_x_ptr = HAS_Z ? out_x.data_ptr<scalar_t>() : nullptr;

        size_t shmem_bytes = 0;

        chunk_scan_fwd_kernel<scalar_t, acc_t><<<grid, 1, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
            cb_ptr, x_ptr, z_ptr, dt_ptr, dA_ptr, seq_ptr, C_ptr, states_ptr, D_ptr,
            out_ptr, out_x_ptr,
            batch, seqlen, nheads, headdim,
            nchunks, chunk_size, ngroups, dstate,
            stride_cb_batch, stride_cb_chunk, stride_cb_group, stride_cb_m, stride_cb_k,
            stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
            stride_dt_batch, stride_dt_head, stride_dt_chunk, stride_dt_csize,
            stride_dA_batch, stride_dA_head, stride_dA_chunk, stride_dA_csize,
            stride_C_batch, stride_C_seqlen, stride_C_group, stride_C_dstate,
            stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
            stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
            HAS_Z, HAS_D, D.defined() && D.dim() == 2, HAS_SEQ_IDX
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
            throw std::runtime_error("CUDA kernel launch failed");
        }
    }));

    if (out_x.defined()) return {out, out_x};
    return {out};
}
