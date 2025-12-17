#ifndef CHUNK_SCAN_CONFIG_H
#define CHUNK_SCAN_CONFIG_H

// Select which optimization step to compile (1-7):
// 1 = Baseline
// 2 = WMMA Tensor Cores
// 3 = Precompute Exp
// 4 = Shared Memory CB
// 5 = Compile-Time Templates
// 6 = Branchless fminf
// 7 = Adaptive Branchless (best of both worlds)

#ifndef CHUNK_SCAN_OPT_LEVEL
#error "CHUNK_SCAN_OPT_LEVEL must be defined (1-7). Set opt_level in kernels_config.json"
#endif

#if CHUNK_SCAN_OPT_LEVEL == 1
#include "01_baseline.cu"
#elif CHUNK_SCAN_OPT_LEVEL == 2
#include "02_wmma_tensor_cores.cu"
#elif CHUNK_SCAN_OPT_LEVEL == 3
#include "03_precompute_exp.cu"
#elif CHUNK_SCAN_OPT_LEVEL == 4
#include "04_shared_memory_cb.cu"
#elif CHUNK_SCAN_OPT_LEVEL == 5
#include "05_compile_time_templates.cu"
#elif CHUNK_SCAN_OPT_LEVEL == 6
#include "06_branchless_fminf.cu"
#elif CHUNK_SCAN_OPT_LEVEL == 7
#include "07_adaptive_branchless.cu"
#else
#error "Invalid CHUNK_SCAN_OPT_LEVEL. Must be 1-7."
#endif

#endif // CHUNK_SCAN_CONFIG_H
