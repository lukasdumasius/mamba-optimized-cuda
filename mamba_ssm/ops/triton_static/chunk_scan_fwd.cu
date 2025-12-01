#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cassert>
#include <tuple>
#include <iostream>
#include <type_traits>


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
    int64_t stride_z_hdim,
    bool HAS_Z, bool HAS_D, bool D_HAS_HDIM, bool HAS_SEQ_IDX, bool IS_CAUSAL, bool USE_VEC2, bool USE_VEC2_Z, bool USE_VEC2_OUT
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
    int seq_start = pid_c * chunk_size;
    int chunk_size_limit = min(chunk_size, max(0, seqlen - seq_start));
    int m_len = min(BLOCK_M, chunk_size_limit - m_start);
    int n_len = min(BLOCK_N, headdim - n_start);

    const int cols_per_thread = USE_VEC2 ? 2 : 1;
    int lane = threadIdx.x;
    int n_limit = n_start + n_len;
    int n_idx0 = n_start + lane * cols_per_thread;
    if (n_idx0 >= n_limit) return;
    int n_idx1 = n_idx0 + 1;
    bool has_col1 = USE_VEC2 && (n_idx1 < n_limit);

    const scalar_t* cb_base = cb + pid_b * stride_cb_batch + pid_c * stride_cb_chunk + group_idx * stride_cb_group;
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
    int64_t seq_idx_prev = 0;
    if (HAS_SEQ_IDX) {
        // seq_idx: (batch, seqlen), assume standard contiguous layout
        seq_idx_base = seq_idx + pid_b * seqlen;
        if (pid_c >= 1) {
            int prev_pos = pid_c * chunk_size - 1;
            if (prev_pos >= 0 && prev_pos < seqlen) {
                seq_idx_prev = seq_idx_base[prev_pos];
            }
        }
    }

    scalar_t* out_base = out + pid_b * stride_out_batch + seq_start * stride_out_seqlen + pid_h * stride_out_head;
    scalar_t* out_x_base = nullptr;
    if (HAS_Z) out_x_base = out_x + pid_b * stride_out_batch + seq_start * stride_out_seqlen + pid_h * stride_out_head;

    // Each thread accumulates over its own head-dim column (n_idx)
    acc_t acc_col0[BLOCK_M];
    acc_t acc_col1[BLOCK_M];
    for (int i = 0; i < m_len; ++i) {
        acc_col0[i] = (acc_t)0;
        if (has_col1) acc_col1[i] = (acc_t)0;
    }

    // Preload dA_cumsum for each output row m and (optionally) seq_idx for scale_m
    acc_t dA_m_arr[BLOCK_M];
    acc_t scale_m_arr[BLOCK_M];
    for (int i = 0; i < m_len; ++i) {
        int m_idx = m_start + i;
        const scalar_t* dAptr = dA_base + m_idx * stride_dA_csize;
        acc_t dA_val = (acc_t)__ldg(dAptr);
        dA_m_arr[i] = dA_val;
        acc_t scale = 0;
        if (!HAS_SEQ_IDX) {
            scale = (acc_t)expf((float)dA_val);
        } else {
            int seq_pos = seq_start + m_idx;
            int64_t seq_m = (seq_pos >= 0 && seq_pos < seqlen) ? seq_idx_base[seq_pos] : (int64_t)-1;
            scale = (seq_m == seq_idx_prev) ? (acc_t)expf((float)dA_val) : (acc_t)0;
        }
        scale_m_arr[i] = scale;
    }

    // compute contribution from C @ prev_states, then apply scale_m per row
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

        // Load prev_states tile for this thread's columns
        acc_t P_tile_col0[BLOCK_D];
        acc_t P_tile_col1[BLOCK_D];
        for (int kk = 0; kk < kd_len; ++kk) {
            int k_idx = kd + kk;
            const scalar_t* pptr0 = states_base + n_idx0 * stride_states_hdim + k_idx * stride_states_dstate;
            scalar_t pv0 = __ldg(pptr0);
            P_tile_col0[kk] = (acc_t)pv0;
            if (has_col1) {
                const scalar_t* pptr1 = states_base + n_idx1 * stride_states_hdim + k_idx * stride_states_dstate;
                scalar_t pv1 = __ldg(pptr1);
                P_tile_col1[kk] = (acc_t)pv1;
            }
        }

        // Multiply: acc += C_tile_reg * P_tile for each column
        for (int i = 0; i < m_len; ++i) {
            acc_t sum0 = acc_col0[i];
            for (int kk = 0; kk < kd_len; ++kk) {
                sum0 += C_tile_reg[i][kk] * P_tile_col0[kk];
            }
            acc_col0[i] = sum0;
            if (has_col1) {
                acc_t sum1 = acc_col1[i];
                for (int kk = 0; kk < kd_len; ++kk) {
                    sum1 += C_tile_reg[i][kk] * P_tile_col1[kk];
                }
                acc_col1[i] = sum1;
            }
        }
    }

    // Apply scale from dA_cumsum per m to C @ prev_states term
    for (int i = 0; i < m_len; ++i) {
        acc_t s = scale_m_arr[i];
        acc_col0[i] *= s;
        if (has_col1) acc_col1[i] *= s;
    }

    // accumlate cb * x over k across chunk dimensions, including dA_cumsum factors and causal masking
    int K_MAX = IS_CAUSAL ? min((pid_m + 1) * BLOCK_M, chunk_size_limit) : chunk_size_limit;
    for (int k0 = 0; k0 < K_MAX; k0 += BLOCK_K) {
        int klen = min(BLOCK_K, K_MAX - k0);

        // load cb_tile into shared memory or registers
        acc_t cb_tile_reg[BLOCK_M][BLOCK_K];
        acc_t x_tile_col0[BLOCK_K];
        acc_t x_tile_col1[BLOCK_K];

        // preload dt_k and dA_k
        acc_t dt_vec[BLOCK_K];
        acc_t dA_k_vec[BLOCK_K];
        for (int kk = 0; kk < klen; ++kk) {
            int k_idx = k0 + kk;
            const scalar_t* dtptr = dt_base + k_idx * stride_dt_csize;
            dt_vec[kk] = (acc_t)__ldg(dtptr);
            const scalar_t* dAkptr = dA_base + k_idx * stride_dA_csize;
            dA_k_vec[kk] = (acc_t)__ldg(dAkptr);
        }

        for (int i = 0; i < m_len; ++i) {
            int m_idx = m_start + i;
            for (int kk = 0; kk < klen; ++kk) {
                int k_idx = k0 + kk;
                // compute mask for causal: require m_idx >= k_idx (within chunk offset) if IS_CAUSAL
                bool pass = (!IS_CAUSAL) || (m_idx >= k_idx); 
                if (!pass) { cb_tile_reg[i][kk] = (acc_t)0; continue; }
                const scalar_t* cbptr = cb_base + m_idx * stride_cb_m + k_idx * stride_cb_k;
                scalar_t v = __ldg(cbptr);
                // multiply by exp(min(dA_m - dA_k, 0)) and dt scalar
                float dA_m = (float)dA_m_arr[i];
                float dA_k = (float)dA_k_vec[kk];
                float diff = dA_m - dA_k;
                if (diff > 0.0f) diff = 0.0f;
                float scale = expf(diff);
                cb_tile_reg[i][kk] = (acc_t)v * dt_vec[kk] * (acc_t)scale;
            }
        }

        // Load x tile for this thread's columns
        for (int kk = 0; kk < klen; ++kk) {
            int k_idx = k0 + kk;
            const scalar_t* base_ptr = x_base + k_idx * stride_x_seqlen + n_idx0 * stride_x_hdim;
            acc_t val0;
            if (USE_VEC2 && has_col1 && stride_x_hdim == 1) {
                const float2* xptr2 = reinterpret_cast<const float2*>(base_ptr);
                float2 xv = __ldg(xptr2);
                val0 = (acc_t)xv.x;
                x_tile_col0[kk] = val0;
                x_tile_col1[kk] = (acc_t)xv.y;
                continue;
            } else {
                val0 = (acc_t)__ldg(base_ptr);
                x_tile_col0[kk] = val0;
                if (has_col1) {
                    const scalar_t* base_ptr1 = x_base + k_idx * stride_x_seqlen + n_idx1 * stride_x_hdim;
                    x_tile_col1[kk] = (acc_t)__ldg(base_ptr1);
                }
            }
        }

        // acc += cb_tile_reg * x_tile for each column
        for (int i = 0; i < m_len; ++i) {
            acc_t sum0 = acc_col0[i];
            for (int kk = 0; kk < klen; ++kk) {
                sum0 += cb_tile_reg[i][kk] * x_tile_col0[kk];
            }
            acc_col0[i] = sum0;
            if (has_col1) {
                acc_t sum1 = acc_col1[i];
                for (int kk = 0; kk < klen; ++kk) {
                    sum1 += cb_tile_reg[i][kk] * x_tile_col1[kk];
                }
                acc_col1[i] = sum1;
            }
        }
    }

    // HAS_D from Triton kernel
    if (HAS_D) {
        for (int i = 0; i < m_len; ++i) {
            int m_idx = m_start + i;
            const scalar_t* xptr0 = x_base + m_idx * stride_x_seqlen + n_idx0 * stride_x_hdim;
            acc_t x_res0 = (acc_t) __ldg(xptr0);
            acc_t Dv0;
            if (D_HAS_HDIM) {
                Dv0 = (acc_t) __ldg(D + pid_h * headdim + n_idx0);
            } else {
                Dv0 = (acc_t) __ldg(D + pid_h);
            }
            acc_col0[i] += x_res0 * Dv0;
            if (has_col1) {
                const scalar_t* xptr1 = x_base + m_idx * stride_x_seqlen + n_idx1 * stride_x_hdim;
                acc_t x_res1 = (acc_t) __ldg(xptr1);
                acc_t Dv1 = D_HAS_HDIM ? (acc_t)__ldg(D + pid_h * headdim + n_idx1)
                                       : (acc_t)__ldg(D + pid_h);
                acc_col1[i] += x_res1 * Dv1;
            }
        }
    }

    // HAS_Z from Triton kernel
    for (int i = 0; i < m_len; ++i) {
        int m_idx = m_start + i;
        acc_t val0 = acc_col0[i];
        acc_t val1 = has_col1 ? acc_col1[i] : (acc_t)0;
        
        // Process z gating for column 0
        if (HAS_Z) {
            out_x_base[m_idx * stride_out_seqlen + n_idx0 * stride_out_hdim] = (scalar_t) val0;
            scalar_t zval0;
            if (USE_VEC2_Z && has_col1) {
                const float2* zptr2 = reinterpret_cast<const float2*>(z_base + m_idx * stride_x_seqlen + n_idx0 * stride_z_hdim);
                float2 zv = __ldg(zptr2);
                zval0 = (scalar_t)zv.x;
                float zf0 = (float)zval0;
                float gated0 = zf0 * (1.0f / (1.0f + expf(-zf0)));
                val0 = val0 * (acc_t)gated0;
                
                // Process column 1 z gating
                if (has_col1) {
                    out_x_base[m_idx * stride_out_seqlen + n_idx1 * stride_out_hdim] = (scalar_t) val1;
                    scalar_t zval1 = (scalar_t)zv.y;
                    float zf1 = (float)zval1;
                    float gated1 = zf1 * (1.0f / (1.0f + expf(-zf1)));
                    val1 = val1 * (acc_t)gated1;
                }
            } else {
                zval0 = __ldg(z_base + m_idx * stride_x_seqlen + n_idx0 * stride_z_hdim);
                float zf0 = (float)zval0;
                float gated0 = zf0 * (1.0f / (1.0f + expf(-zf0)));
                val0 = val0 * (acc_t)gated0;
                
                if (has_col1) {
                    out_x_base[m_idx * stride_out_seqlen + n_idx1 * stride_out_hdim] = (scalar_t) val1;
                    scalar_t zval1 = __ldg(z_base + m_idx * stride_x_seqlen + n_idx1 * stride_z_hdim);
                    float zf1 = (float)zval1;
                    float gated1 = zf1 * (1.0f / (1.0f + expf(-zf1)));
                    val1 = val1 * (acc_t)gated1;
                }
            }
        }
        
        // Write outputs with vectorization if possible
        if (USE_VEC2_OUT && has_col1) {
            float2* outptr2 = reinterpret_cast<float2*>(out_base + m_idx * stride_out_seqlen + n_idx0 * stride_out_hdim);
            float2 outv;
            outv.x = (float)val0;
            outv.y = (float)val1;
            *outptr2 = outv;
        } else {
            out_base[m_idx * stride_out_seqlen + n_idx0 * stride_out_hdim] = (scalar_t) val0;
            if (has_col1) {
                out_base[m_idx * stride_out_seqlen + n_idx1 * stride_out_hdim] = (scalar_t) val1;
            }
        }
    }
}

// Host wrapper used by Python binding `chunk_scan_fwd_cuda`
std::vector<torch::Tensor> chunk_scan_fwd_cuda(
    torch::Tensor cb,
    torch::Tensor x,
    torch::Tensor dt,
    torch::Tensor dA_cumsum,
    torch::Tensor C,
    torch::Tensor states,
    c10::optional<torch::Tensor> D_opt,
    c10::optional<torch::Tensor> z_opt,
    c10::optional<torch::Tensor> seq_idx_opt
) {
    // Unpack optionals
    torch::Tensor D = D_opt.has_value() ? *D_opt : torch::Tensor();
    torch::Tensor z = z_opt.has_value() ? *z_opt : torch::Tensor();
    torch::Tensor seq_idx = seq_idx_opt.has_value() ? *seq_idx_opt : torch::Tensor();
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

    // IS_CAUSAL is always true in current usage (`chunk_scan` is causal)
    bool IS_CAUSAL = true;

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

        constexpr bool kIsFloat = std::is_same<scalar_t, float>::value;
        // Always enable vectorization for fp32 (handle odd headdim with special case)
        bool use_vec2 = kIsFloat && (stride_x_hdim == 1);
        // Check if z and out can be vectorized (stride must be 1)
        int64_t stride_z_hdim = HAS_Z ? z.strides()[3] : 0;
        bool use_vec2_z = kIsFloat && HAS_Z && (stride_z_hdim == 1);
        bool use_vec2_out = kIsFloat && (stride_out_hdim == 1);
        int threads_x = use_vec2 ? (BLOCK_N / 2) : BLOCK_N;
        dim3 block(threads_x);

        chunk_scan_fwd_kernel<scalar_t, acc_t><<<grid, block, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
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
            stride_z_hdim,
            HAS_Z, HAS_D, D.defined() && D.dim() == 2, HAS_SEQ_IDX, IS_CAUSAL, use_vec2, use_vec2_z, use_vec2_out
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
