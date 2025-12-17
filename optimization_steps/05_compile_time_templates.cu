#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cassert>
#include <tuple>
#include <iostream>
#include <type_traits>
#include <mma.h>
#include <cuda_pipeline.h>

__host__ __device__ inline int div_up(int a, int b) { return (a + b - 1) / b; }

#define BLOCK_M 16
#define BLOCK_N 64
#define BLOCK_K 32
#define BLOCK_D 64
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

template<typename scalar_t, 
    typename acc_t,
    bool HAS_Z,
    bool HAS_D,
    bool D_HAS_HDIM,
    bool HAS_SEQ_IDX,
    bool IS_CAUSAL,
    bool USE_VEC2,
    bool USE_VEC2_Z,
    bool USE_VEC2_OUT>
__global__ void chunk_scan_fwd_kernel(
    const scalar_t* __restrict__ cb,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ z,
    const scalar_t* __restrict__ dt,
    const float* __restrict__ dA_cumsum,
    const int64_t* __restrict__ seq_idx,
    const scalar_t* __restrict__ C,
    const scalar_t* __restrict__ states,
    const scalar_t* __restrict__ D,
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
    int64_t stride_z_hdim
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
    int chunk_size_lim = min(chunk_size, max(0, seqlen - seq_start));
    int m_len = min(BLOCK_M, chunk_size_lim - m_start);
    int n_len = min(BLOCK_N, headdim - n_start);

    if (m_len <= 0 || n_len <= 0) return;

    const int cols_per_thread = USE_VEC2 ? 2 : 1;
    int lane = threadIdx.x;
    int n_limit = n_start + n_len;
    int n_idx0 = n_start + lane * cols_per_thread;
    int n_idx1 = n_idx0 + 1;
    bool has_col0 = (n_idx0 < n_limit);
    bool has_col1 = (cols_per_thread == 2) && (n_idx1 < n_limit);

    const scalar_t* cb_base = cb + pid_b * stride_cb_batch + pid_c * stride_cb_chunk + group_idx * stride_cb_group;
    const scalar_t* x_base = x + pid_b * stride_x_batch + seq_start * stride_x_seqlen + pid_h * stride_x_head;
    const scalar_t* dt_base = dt + pid_b * stride_dt_batch + pid_h * stride_dt_head + pid_c * stride_dt_chunk;
    const float* dA_base = dA_cumsum + pid_b * stride_dA_batch + pid_h * stride_dA_head + pid_c * stride_dA_chunk;
    const scalar_t* C_base = C + pid_b * stride_C_batch + seq_start * stride_C_seqlen + group_idx * stride_C_group;
    const scalar_t* states_base = states + pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head;

    const scalar_t* z_base = nullptr;
    if constexpr (HAS_Z) {
        z_base = z + pid_b * stride_x_batch + seq_start * stride_x_seqlen + pid_h * stride_x_head;
    }

    scalar_t* out_base = out + pid_b * stride_out_batch + seq_start * stride_out_seqlen + pid_h * stride_out_head;
    scalar_t* out_x_base = nullptr;
    if constexpr (HAS_Z) {
        out_x_base = out_x + pid_b * stride_out_batch + seq_start * stride_out_seqlen + pid_h * stride_out_head;
    }

    const int64_t* seq_idx_base = nullptr;
    int64_t seq_idx_prev = 0;
    if constexpr (HAS_SEQ_IDX) {
        seq_idx_base = seq_idx + pid_b * seqlen;
        if (pid_c >= 1) {
            int prev_pos = pid_c * chunk_size - 1;
            int prev_pos_actual = min(prev_pos, seqlen - 1);
            if (prev_pos_actual >= 0) seq_idx_prev = seq_idx_base[prev_pos_actual];
        }
    }

    acc_t dA_m_arr[BLOCK_M];
    acc_t scale_m_arr[BLOCK_M];
    float expf_dA_m[BLOCK_M];

    if (has_col0) {
        for (int i = 0; i < m_len; ++i) {
            int m_idx = m_start + i;
            const float* dAptr = dA_base + m_idx * stride_dA_csize;
            float dA_val_f = __ldg(dAptr);
            dA_m_arr[i] = (acc_t)dA_val_f;
            expf_dA_m[i] = expf(dA_val_f);
            acc_t scale;
            if constexpr (!HAS_SEQ_IDX) {
                scale = (acc_t)expf_dA_m[i];
            } else {
                int seq_pos = seq_start + m_idx;
                int64_t seq_m = (seq_pos >= 0 && seq_pos < seqlen) ? seq_idx_base[seq_pos] : (int64_t)-1;
                scale = (seq_m == seq_idx_prev) ? (acc_t)expf_dA_m[i] : (acc_t)0;
            }
            scale_m_arr[i] = scale;
        }
    }

    extern __shared__ char smem_pool[];
    __half* smem_C = reinterpret_cast<__half*>(smem_pool);
    __half* smem_States = reinterpret_cast<__half*>(smem_pool + BLOCK_M * (BLOCK_D + 8) * sizeof(__half));
    float* smem_Accum = reinterpret_cast<float*>(smem_pool + BLOCK_M * (BLOCK_D + 8) * sizeof(__half) + BLOCK_D * (BLOCK_N + 8) * sizeof(__half));
    __half* smem_cb = reinterpret_cast<__half*>(smem_pool + BLOCK_M * (BLOCK_D + 8) * sizeof(__half) + BLOCK_D * (BLOCK_N + 8) * sizeof(__half) + BLOCK_M * (BLOCK_N + 8) * sizeof(float));

    int tid = threadIdx.x;
    int warpId = tid >> 5;
    int num_warps = blockDim.x >> 5;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frags[4];

    for (int kd = 0; kd < dstate; kd += BLOCK_D) {
        for (int i = tid; i < BLOCK_M * BLOCK_D; i += blockDim.x) {
            int r = i / BLOCK_D;
            int c = i % BLOCK_D;
            int smem_idx = r * (BLOCK_D + 8) + c;
            int global_m = m_start + r;
            int global_k = kd + c;
            if (global_m < chunk_size_lim && global_k < dstate) {
                const scalar_t* src = C_base + global_m * stride_C_seqlen + global_k * stride_C_dstate;
                smem_C[smem_idx] = (__half)(*src);
            } else {
                smem_C[smem_idx] = __half(0.0f);
            }
        }

        for (int i = tid; i < BLOCK_D * BLOCK_N; i += blockDim.x) {
            int r = i / BLOCK_N;
            int c = i % BLOCK_N;
            int smem_idx = r * (BLOCK_N + 8) + c;
            int global_k = kd + r;
            int global_n = n_start + c;
            if (global_k < dstate && global_n < headdim) {
                const scalar_t* src = states_base + global_n * stride_states_hdim + global_k * stride_states_dstate;
                smem_States[smem_idx] = (__half)(*src);
            } else {
                smem_States[smem_idx] = __half(0.0f);
            }
        }

        __syncthreads();

        for (int w_m = warpId * WMMA_M; w_m < BLOCK_M; w_m += num_warps * WMMA_M) {
            if (w_m >= BLOCK_M) break;
            int m_tile_end = w_m + WMMA_M;
            if (m_tile_end <= 0 || w_m >= m_len) continue;

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                nvcuda::wmma::fill_fragment(acc_frags[i], 0.0f);
            }

            for (int k_step = 0; k_step < BLOCK_D; k_step += WMMA_K) {
                int a_off = w_m * (BLOCK_D + 8) + k_step;
                nvcuda::wmma::load_matrix_sync(a_frag, smem_C + a_off, BLOCK_D + 8);

                #pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    int n_step = ni * WMMA_N;
                    int b_off = k_step * (BLOCK_N + 8) + n_step;
                    nvcuda::wmma::load_matrix_sync(b_frag, smem_States + b_off, BLOCK_N + 8);
                    nvcuda::wmma::mma_sync(acc_frags[ni], a_frag, b_frag, acc_frags[ni]);
                }
            }

            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                int n_step = ni * WMMA_N;
                float* dest = smem_Accum + w_m * (BLOCK_N + 8) + n_step;
                nvcuda::wmma::store_matrix_sync(dest, acc_frags[ni], BLOCK_N + 8, nvcuda::wmma::mem_row_major);
            }
        }

        __syncthreads();
    }

    acc_t acc_col0[BLOCK_M];
    acc_t acc_col1[BLOCK_M];

    if (has_col0) {
        int local_n0 = n_idx0 - n_start;
        int local_n1 = n_idx1 - n_start;
        for (int i = 0; i < m_len; ++i) {
            acc_t base0 = (local_n0 >= 0 && local_n0 < BLOCK_N) ? (acc_t)smem_Accum[i * (BLOCK_N + 8) + local_n0] : (acc_t)0;
            acc_col0[i] = base0 * scale_m_arr[i];
            if (has_col1) {
                acc_t base1 = (local_n1 >= 0 && local_n1 < BLOCK_N) ? (acc_t)smem_Accum[i * (BLOCK_N + 8) + local_n1] : (acc_t)0;
                acc_col1[i] = base1 * scale_m_arr[i];
            }
        }
    } else {
        for (int i = 0; i < m_len; ++i) {
            acc_col0[i] = (acc_t)0;
            if (has_col1) acc_col1[i] = (acc_t)0;
        }
    }

    __syncthreads();

    int K_MAX = IS_CAUSAL ? min((pid_m + 1) * BLOCK_M, chunk_size_lim) : chunk_size_lim;

    if (has_col0) {
        for (int k0 = 0; k0 < K_MAX; k0 += BLOCK_K) {
            int klen = min(BLOCK_K, K_MAX - k0);

            if (lane < m_len) {
                int i = lane;
                int m_idx = m_start + i;
                for (int kk = 0; kk < klen; ++kk) {
                    int k_idx = k0 + kk;
                    if (k_idx < K_MAX) {
                        const scalar_t* cbptr = cb_base + m_idx * stride_cb_m + k_idx * stride_cb_k;
                        smem_cb[i * (BLOCK_K + 8) + kk] = (__half)__ldg(cbptr);
                    } else {
                        smem_cb[i * (BLOCK_K + 8) + kk] = (__half)0.0f;
                    }
                }
            }
            __syncthreads();

            acc_t dt_vec[BLOCK_K];
            float expf_neg_dA_k[BLOCK_K];
            acc_t x_tile_col0[BLOCK_K];
            acc_t x_tile_col1[BLOCK_K];

            for (int kk = 0; kk < klen; ++kk) {
                int k_idx = k0 + kk;
                const scalar_t* dtptr = dt_base + k_idx * stride_dt_csize;
                dt_vec[kk] = (acc_t)__ldg(dtptr);
                const float* dAkptr = dA_base + k_idx * stride_dA_csize;
                float dAk_f = __ldg(dAkptr);
                expf_neg_dA_k[kk] = expf(-dAk_f);

                const scalar_t* base_ptr0 = x_base + k_idx * stride_x_seqlen + n_idx0 * stride_x_hdim;
                if constexpr (USE_VEC2) {
                    if (has_col1 && stride_x_hdim == 1) {
                        const float2* xptr2 = reinterpret_cast<const float2*>(base_ptr0);
                        float2 xv = __ldg(xptr2);
                        x_tile_col0[kk] = (acc_t)xv.x;
                        x_tile_col1[kk] = (acc_t)xv.y;
                    } else {
                        x_tile_col0[kk] = (acc_t)__ldg(base_ptr0);
                        if (has_col1) {
                            const scalar_t* base_ptr1 = x_base + k_idx * stride_x_seqlen + n_idx1 * stride_x_hdim;
                            x_tile_col1[kk] = (acc_t)__ldg(base_ptr1);
                        }
                    }
                } else {
                    x_tile_col0[kk] = (acc_t)__ldg(base_ptr0);
                    if (has_col1) {
                        const scalar_t* base_ptr1 = x_base + k_idx * stride_x_seqlen + n_idx1 * stride_x_hdim;
                        x_tile_col1[kk] = (acc_t)__ldg(base_ptr1);
                    }
                }
            }

            for (int i = 0; i < m_len; ++i) {
                int m_idx = m_start + i;
                acc_t sum0 = 0;
                acc_t sum1 = 0;
                int k_end = IS_CAUSAL ? min(klen, max(0, m_idx - k0 + 1)) : klen;

                for (int kk = 0; kk < k_end; ++kk) {
                    __half v_half = smem_cb[i * (BLOCK_K + 8) + kk];
                    scalar_t v = (scalar_t)v_half;

                    float dA_mf = (float)dA_m_arr[i];
                    float scale;
                    if (dA_mf <= 0.0f) {
                        scale = expf_dA_m[i] * expf_neg_dA_k[kk];
                    } else {
                        scale = 1.0f;
                    }

                    acc_t cb_val = (acc_t)v * dt_vec[kk] * (acc_t)scale;
                    sum0 += cb_val * x_tile_col0[kk];
                    if (has_col1) sum1 += cb_val * x_tile_col1[kk];
                }
                acc_col0[i] += sum0;
                if (has_col1) acc_col1[i] += sum1;
            }

            __syncthreads();
        }
    }

    if (!has_col0) return;

    if constexpr (HAS_D) {
        for (int i = 0; i < m_len; ++i) {
            int m_idx = m_start + i;
            const scalar_t* xptr0 = x_base + m_idx * stride_x_seqlen + n_idx0 * stride_x_hdim;
            acc_t x_res0 = (acc_t)__ldg(xptr0);
            acc_t Dv0;
            if constexpr (D_HAS_HDIM) {
                Dv0 = (acc_t)__ldg(D + pid_h * headdim + n_idx0);
            } else {
                Dv0 = (acc_t)__ldg(D + pid_h);
            }
            acc_col0[i] += x_res0 * Dv0;
            if (has_col1) {
                const scalar_t* xptr1 = x_base + m_idx * stride_x_seqlen + n_idx1 * stride_x_hdim;
                acc_t x_res1 = (acc_t)__ldg(xptr1);
                acc_t Dv1;
                if constexpr (D_HAS_HDIM) {
                    Dv1 = (acc_t)__ldg(D + pid_h * headdim + n_idx1);
                } else {
                    Dv1 = (acc_t)__ldg(D + pid_h);
                }
                acc_col1[i] += x_res1 * Dv1;
            }
        }
    }

    for (int i = 0; i < m_len; ++i) {
        int m_idx = m_start + i;
        acc_t val0 = acc_col0[i];
        acc_t val1 = has_col1 ? acc_col1[i] : (acc_t)0;

        if constexpr (HAS_Z) {
            out_x_base[m_idx * stride_out_seqlen + n_idx0 * stride_out_hdim] = (scalar_t)val0;
            if constexpr (USE_VEC2_Z) {
                if (has_col1) {
                    const float2* zptr2 = reinterpret_cast<const float2*>(z_base + m_idx * stride_x_seqlen + n_idx0 * stride_z_hdim);
                    float2 zv = __ldg(zptr2);
                    float zf0 = (float)(scalar_t)zv.x;
                    float gated0 = zf0 * (1.0f / (1.0f + expf(-zf0)));
                    val0 *= (acc_t)gated0;
                    out_x_base[m_idx * stride_out_seqlen + n_idx1 * stride_out_hdim] = (scalar_t)val1;
                    float zf1 = (float)(scalar_t)zv.y;
                    float gated1 = zf1 * (1.0f / (1.0f + expf(-zf1)));
                    val1 *= (acc_t)gated1;
                } else {
                    scalar_t z0 = __ldg(z_base + m_idx * stride_x_seqlen + n_idx0 * stride_z_hdim);
                    float zf0 = (float)z0;
                    float gated0 = zf0 * (1.0f / (1.0f + expf(-zf0)));
                    val0 *= (acc_t)gated0;
                }
            } else {
                scalar_t z0 = __ldg(z_base + m_idx * stride_x_seqlen + n_idx0 * stride_z_hdim);
                float zf0 = (float)z0;
                float gated0 = zf0 * (1.0f / (1.0f + expf(-zf0)));
                val0 *= (acc_t)gated0;
                if (has_col1) {
                    out_x_base[m_idx * stride_out_seqlen + n_idx1 * stride_out_hdim] = (scalar_t)val1;
                    scalar_t z1 = __ldg(z_base + m_idx * stride_x_seqlen + n_idx1 * stride_z_hdim);
                    float zf1 = (float)z1;
                    float gated1 = zf1 * (1.0f / (1.0f + expf(-zf1)));
                    val1 *= (acc_t)gated1;
                }
            }
        }

        if constexpr (USE_VEC2_OUT) {
            if (has_col1) {
                float2* outptr2 = reinterpret_cast<float2*>(out_base + m_idx * stride_out_seqlen + n_idx0 * stride_out_hdim);
                float2 outv;
                outv.x = (float)val0;
                outv.y = (float)val1;
                *outptr2 = outv;
            } else {
                out_base[m_idx * stride_out_seqlen + n_idx0 * stride_out_hdim] = (scalar_t)val0;
            }
        } else {
            out_base[m_idx * stride_out_seqlen + n_idx0 * stride_out_hdim] = (scalar_t)val0;
            if (has_col1) {
                out_base[m_idx * stride_out_seqlen + n_idx1 * stride_out_hdim] = (scalar_t)val1;
            }
        }
    }
}

template<typename scalar_t, bool HAS_Z, bool HAS_D, bool D_HAS_HDIM, bool HAS_SEQ_IDX, bool USE_VEC2, bool USE_VEC2_Z, bool USE_VEC2_OUT>
inline void launch_chunk_scan_fwd_kernel(
    dim3 grid, dim3 block, size_t shmem_bytes, cudaStream_t stream,
    const scalar_t* cb_ptr, const scalar_t* x_ptr, const scalar_t* z_ptr,
    const scalar_t* dt_ptr, const float* dA_ptr, const int64_t* seq_ptr,
    const scalar_t* C_ptr, const scalar_t* states_ptr, const scalar_t* D_ptr,
    scalar_t* out_ptr, scalar_t* out_x_ptr,
    int batch, int seqlen, int nheads, int headdim, int nchunks, int chunk_size, int ngroups, int dstate,
    int64_t stride_cb_batch, int64_t stride_cb_chunk, int64_t stride_cb_group, int64_t stride_cb_m, int64_t stride_cb_k,
    int64_t stride_x_batch, int64_t stride_x_seqlen, int64_t stride_x_head, int64_t stride_x_hdim,
    int64_t stride_dt_batch, int64_t stride_dt_head, int64_t stride_dt_chunk, int64_t stride_dt_csize,
    int64_t stride_dA_batch, int64_t stride_dA_head, int64_t stride_dA_chunk, int64_t stride_dA_csize,
    int64_t stride_C_batch, int64_t stride_C_seqlen, int64_t stride_C_group, int64_t stride_C_dstate,
    int64_t stride_states_batch, int64_t stride_states_chunk, int64_t stride_states_head, int64_t stride_states_hdim, int64_t stride_states_dstate,
    int64_t stride_out_batch, int64_t stride_out_seqlen, int64_t stride_out_head, int64_t stride_out_hdim,
    int64_t stride_z_hdim
) {
    using acc_t = float;
    constexpr bool IS_CAUSAL = true;
    chunk_scan_fwd_kernel<scalar_t, acc_t, HAS_Z, HAS_D, D_HAS_HDIM, HAS_SEQ_IDX, IS_CAUSAL, USE_VEC2, USE_VEC2_Z, USE_VEC2_OUT>
        <<<grid, block, shmem_bytes, stream>>>(
            cb_ptr, x_ptr, z_ptr, dt_ptr, dA_ptr, seq_ptr, C_ptr, states_ptr, D_ptr,
            out_ptr, out_x_ptr, batch, seqlen, nheads, headdim, nchunks, chunk_size, ngroups, dstate,
            stride_cb_batch, stride_cb_chunk, stride_cb_group, stride_cb_m, stride_cb_k,
            stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
            stride_dt_batch, stride_dt_head, stride_dt_chunk, stride_dt_csize,
            stride_dA_batch, stride_dA_head, stride_dA_chunk, stride_dA_csize,
            stride_C_batch, stride_C_seqlen, stride_C_group, stride_C_dstate,
            stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
            stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim, stride_z_hdim
        );
}

template<typename scalar_t, bool HAS_Z, bool HAS_D, bool D_HAS_HDIM, bool HAS_SEQ_IDX>
inline void dispatch_vec_variant(
    bool vec_all, dim3 grid, dim3 block, size_t shmem_bytes, cudaStream_t stream,
    const scalar_t* cb_ptr, const scalar_t* x_ptr, const scalar_t* z_ptr,
    const scalar_t* dt_ptr, const float* dA_ptr, const int64_t* seq_ptr,
    const scalar_t* C_ptr, const scalar_t* states_ptr, const scalar_t* D_ptr,
    scalar_t* out_ptr, scalar_t* out_x_ptr,
    int batch, int seqlen, int nheads, int headdim, int nchunks, int chunk_size, int ngroups, int dstate,
    int64_t stride_cb_batch, int64_t stride_cb_chunk, int64_t stride_cb_group, int64_t stride_cb_m, int64_t stride_cb_k,
    int64_t stride_x_batch, int64_t stride_x_seqlen, int64_t stride_x_head, int64_t stride_x_hdim,
    int64_t stride_dt_batch, int64_t stride_dt_head, int64_t stride_dt_chunk, int64_t stride_dt_csize,
    int64_t stride_dA_batch, int64_t stride_dA_head, int64_t stride_dA_chunk, int64_t stride_dA_csize,
    int64_t stride_C_batch, int64_t stride_C_seqlen, int64_t stride_C_group, int64_t stride_C_dstate,
    int64_t stride_states_batch, int64_t stride_states_chunk, int64_t stride_states_head, int64_t stride_states_hdim, int64_t stride_states_dstate,
    int64_t stride_out_batch, int64_t stride_out_seqlen, int64_t stride_out_head, int64_t stride_out_hdim,
    int64_t stride_z_hdim
) {
    if (vec_all) {
        launch_chunk_scan_fwd_kernel<scalar_t, HAS_Z, HAS_D, D_HAS_HDIM, HAS_SEQ_IDX, true, true, true>(
            grid, block, shmem_bytes, stream, cb_ptr, x_ptr, z_ptr, dt_ptr, dA_ptr, seq_ptr,
            C_ptr, states_ptr, D_ptr, out_ptr, out_x_ptr, batch, seqlen, nheads, headdim, nchunks, chunk_size, ngroups, dstate,
            stride_cb_batch, stride_cb_chunk, stride_cb_group, stride_cb_m, stride_cb_k,
            stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
            stride_dt_batch, stride_dt_head, stride_dt_chunk, stride_dt_csize,
            stride_dA_batch, stride_dA_head, stride_dA_chunk, stride_dA_csize,
            stride_C_batch, stride_C_seqlen, stride_C_group, stride_C_dstate,
            stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
            stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim, stride_z_hdim);
    } else {
        launch_chunk_scan_fwd_kernel<scalar_t, HAS_Z, HAS_D, D_HAS_HDIM, HAS_SEQ_IDX, false, false, false>(
            grid, block, shmem_bytes, stream, cb_ptr, x_ptr, z_ptr, dt_ptr, dA_ptr, seq_ptr,
            C_ptr, states_ptr, D_ptr, out_ptr, out_x_ptr, batch, seqlen, nheads, headdim, nchunks, chunk_size, ngroups, dstate,
            stride_cb_batch, stride_cb_chunk, stride_cb_group, stride_cb_m, stride_cb_k,
            stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
            stride_dt_batch, stride_dt_head, stride_dt_chunk, stride_dt_csize,
            stride_dA_batch, stride_dA_head, stride_dA_chunk, stride_dA_csize,
            stride_C_batch, stride_C_seqlen, stride_C_group, stride_C_dstate,
            stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
            stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim, stride_z_hdim);
    }
}

std::vector<torch::Tensor> chunk_scan_fwd_cuda(
    torch::Tensor cb, torch::Tensor x, torch::Tensor dt, torch::Tensor dA_cumsum,
    torch::Tensor C, torch::Tensor states,
    c10::optional<torch::Tensor> D_opt, c10::optional<torch::Tensor> z_opt,
    c10::optional<torch::Tensor> seq_idx_opt
) {
    torch::Tensor D = D_opt.has_value() ? *D_opt : torch::Tensor();
    torch::Tensor z = z_opt.has_value() ? *z_opt : torch::Tensor();
    torch::Tensor seq_idx = seq_idx_opt.has_value() ? *seq_idx_opt : torch::Tensor();
    bool HAS_Z = z.defined();
    bool HAS_D = D.defined();
    bool HAS_SEQ_IDX = seq_idx.defined();

    int batch = (int)cb.size(0), nchunks = (int)cb.size(1), ngroups = (int)cb.size(2);
    int chunk_size = (int)cb.size(3), seqlen = (int)x.size(1);
    int nheads = (int)x.size(2), headdim = (int)x.size(3), dstate = (int)C.size(3);

    auto out = torch::empty_like(x);
    torch::Tensor out_x = HAS_Z ? torch::empty_like(x) : torch::Tensor();

    auto cb_strides = cb.strides();
    auto x_strides = x.strides();
    auto dt_strides = dt.strides();
    auto dA_strides = dA_cumsum.strides();
    auto C_strides = C.strides();
    auto states_strides = states.strides();
    auto out_strides = out.strides();
    auto z_strides = HAS_Z ? z.strides() : x.strides();

    int num_pid_m = div_up(chunk_size, BLOCK_M);
    int num_pid_n = div_up(headdim, BLOCK_N);
    dim3 grid(num_pid_m * num_pid_n, batch * nchunks, nheads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "chunk_scan_fwd_cuda", [&] {
        using scalar_t = scalar_t;

        size_t smem_C_bytes = BLOCK_M * (BLOCK_D + 8) * sizeof(__half);
        size_t smem_States_bytes = BLOCK_D * (BLOCK_N + 8) * sizeof(__half);
        size_t smem_Accum_bytes = BLOCK_M * (BLOCK_N + 8) * sizeof(float);
        size_t smem_cb_bytes = BLOCK_M * (BLOCK_K + 8) * sizeof(__half);
        size_t shmem_bytes = smem_C_bytes + smem_States_bytes + smem_Accum_bytes + smem_cb_bytes;

        constexpr bool kIsFloat = std::is_same<scalar_t, float>::value;
        int64_t stride_z_hdim = HAS_Z ? z_strides[3] : 0;
        bool can_vec2 = kIsFloat && (x_strides[3] == 1);
        bool can_vec2_z = kIsFloat && HAS_Z && (stride_z_hdim == 1);
        bool can_vec2_out = kIsFloat && (out_strides[3] == 1);
        bool vec_all = can_vec2 && can_vec2_z && can_vec2_out;

        dim3 block(64);
        auto stream = at::cuda::getCurrentCUDAStream();
        bool d_has_hdim = HAS_D && D.dim() == 2;

        const scalar_t* cb_ptr = cb.data_ptr<scalar_t>();
        const scalar_t* x_ptr = x.data_ptr<scalar_t>();
        const scalar_t* z_ptr = HAS_Z ? z.data_ptr<scalar_t>() : nullptr;
        const scalar_t* dt_ptr = dt.data_ptr<scalar_t>();
        const float* dA_ptr = dA_cumsum.data_ptr<float>();
        const int64_t* seq_ptr = HAS_SEQ_IDX ? seq_idx.data_ptr<int64_t>() : nullptr;
        const scalar_t* C_ptr = C.data_ptr<scalar_t>();
        const scalar_t* states_ptr = states.data_ptr<scalar_t>();
        const scalar_t* D_ptr = HAS_D ? D.data_ptr<scalar_t>() : nullptr;
        scalar_t* out_ptr = out.data_ptr<scalar_t>();
        scalar_t* out_x_ptr = HAS_Z ? out_x.data_ptr<scalar_t>() : nullptr;

        if (HAS_Z && HAS_D && HAS_SEQ_IDX && d_has_hdim) {
            dispatch_vec_variant<scalar_t, true, true, true, true>(vec_all, grid, block, shmem_bytes, stream,
                cb_ptr, x_ptr, z_ptr, dt_ptr, dA_ptr, seq_ptr, C_ptr, states_ptr, D_ptr, out_ptr, out_x_ptr,
                batch, seqlen, nheads, headdim, nchunks, chunk_size, ngroups, dstate,
                cb_strides[0], cb_strides[1], cb_strides[2], cb_strides[3], cb_strides[4],
                x_strides[0], x_strides[1], x_strides[2], x_strides[3],
                dt_strides[0], dt_strides[1], dt_strides[2], dt_strides[3],
                dA_strides[0], dA_strides[1], dA_strides[2], dA_strides[3],
                C_strides[0], C_strides[1], C_strides[2], C_strides[3],
                states_strides[0], states_strides[1], states_strides[2], states_strides[3], states_strides[4],
                out_strides[0], out_strides[1], out_strides[2], out_strides[3], stride_z_hdim);
        } else if (!HAS_Z && !HAS_D && !HAS_SEQ_IDX) {
            dispatch_vec_variant<scalar_t, false, false, false, false>(vec_all, grid, block, shmem_bytes, stream,
                cb_ptr, x_ptr, z_ptr, dt_ptr, dA_ptr, seq_ptr, C_ptr, states_ptr, D_ptr, out_ptr, out_x_ptr,
                batch, seqlen, nheads, headdim, nchunks, chunk_size, ngroups, dstate,
                cb_strides[0], cb_strides[1], cb_strides[2], cb_strides[3], cb_strides[4],
                x_strides[0], x_strides[1], x_strides[2], x_strides[3],
                dt_strides[0], dt_strides[1], dt_strides[2], dt_strides[3],
                dA_strides[0], dA_strides[1], dA_strides[2], dA_strides[3],
                C_strides[0], C_strides[1], C_strides[2], C_strides[3],
                states_strides[0], states_strides[1], states_strides[2], states_strides[3], states_strides[4],
                out_strides[0], out_strides[1], out_strides[2], out_strides[3], stride_z_hdim);
        } else {
            dispatch_vec_variant<scalar_t, true, true, true, true>(vec_all, grid, block, shmem_bytes, stream,
                cb_ptr, x_ptr, z_ptr, dt_ptr, dA_ptr, seq_ptr, C_ptr, states_ptr, D_ptr, out_ptr, out_x_ptr,
                batch, seqlen, nheads, headdim, nchunks, chunk_size, ngroups, dstate,
                cb_strides[0], cb_strides[1], cb_strides[2], cb_strides[3], cb_strides[4],
                x_strides[0], x_strides[1], x_strides[2], x_strides[3],
                dt_strides[0], dt_strides[1], dt_strides[2], dt_strides[3],
                dA_strides[0], dA_strides[1], dA_strides[2], dA_strides[3],
                C_strides[0], C_strides[1], C_strides[2], C_strides[3],
                states_strides[0], states_strides[1], states_strides[2], states_strides[3], states_strides[4],
                out_strides[0], out_strides[1], out_strides[2], out_strides[3], stride_z_hdim);
        }
    });

    if (out_x.defined()) return {out, out_x};
    return {out};
}



