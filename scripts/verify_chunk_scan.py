#!/usr/bin/env python3
import torch
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Toggle which test harness to run
USE_ENHANCED_TEST = True


def benchmark_kernel(func, *args, warmup=10, repeat=100):
    """Benchmark a kernel function with warmup and multiple runs."""
    # Warmup
    for _ in range(warmup):
        _ = func(*args)
    
    # Synchronize before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Time multiple runs
    start = time.perf_counter()
    for _ in range(repeat):
        _ = func(*args)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / repeat * 1000  # Return average time in milliseconds


def test_chunk_scan():
    from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd

    print("Testing chunk_scan_fwd kernel (Triton vs CUDA)...")
    device = "cuda"

    # Test configurations: (batch, seqlen, nheads, hdim, ngroups, dstate, chunk_size)
    configs = [
        (1, 64, 4, 32, 1, 16, 64),
        (2, 128, 4, 64, 2, 16, 64),
        (1, 256, 8, 64, 1, 32, 64),
        (2, 256, 8, 64, 2, 32, 128),
    ]
    dtypes = [torch.float16]  # Only FP16 - CUDA kernel is WMMA-only

    passed = 0
    total = 0

    for batch, seqlen, nheads, hdim, ngroups, dstate, chunk_size in configs:
        for dtype in dtypes:
            total += 3  # we run three flag combinations per config/dtype
            try:
                nchunks = (seqlen + chunk_size - 1) // chunk_size

                # Generate inputs
                cb = torch.randn(batch, nchunks, ngroups, chunk_size, chunk_size,
                                 device=device, dtype=dtype) * 0.1
                x = torch.randn(batch, seqlen, nheads, hdim, device=device, dtype=dtype)
                dt = torch.rand(batch, nheads, nchunks, chunk_size, device=device, dtype=dtype) * 0.1 + 0.01
                dA_cumsum = torch.randn(batch, nheads, nchunks, chunk_size, device=device, dtype=torch.float32) * 0.5
                C = torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=dtype) * 0.1
                states = torch.randn(batch, nchunks, nheads, hdim, dstate, device=device, dtype=dtype) * 0.1
                D = torch.randn(nheads, hdim, device=device, dtype=dtype) * 0.1
                z = torch.randn(batch, seqlen, nheads, hdim, device=device, dtype=dtype)

                # Simple seq_idx pattern to exercise segmentation logic across chunks
                seq_idx = torch.zeros(batch, seqlen, device=device, dtype=torch.long)
                half = seqlen // 2
                if half > 0:
                    seq_idx[:, half:] = 1

                # Clone inputs for reference
                cb_ref = cb.clone()
                x_ref = x.clone()
                dt_ref = dt.clone()
                dA_cumsum_ref = dA_cumsum.clone()
                C_ref = C.clone()
                states_ref = states.clone()
                D_ref = D.clone()
                z_ref = z.clone()
                seq_idx_ref = seq_idx.clone()

                config_path = Path(__file__).parent.parent / "kernels_config.json"
                with open(config_path) as f:
                    config = json.load(f)

                # Helper to run one comparison under specific feature flags
                def run_case(use_D: bool, use_z: bool, use_seq_idx: bool):
                    nonlocal passed
                    # Select which tensors to pass
                    D_t = D_ref if use_D else None
                    z_t = z_ref if use_z else None
                    seq_t = seq_idx_ref if use_seq_idx else None

                    # Run Triton kernel
                    config["ssd_chunk_scan"]["use_cuda"] = False
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)

                    import importlib
                    import mamba_ssm.ops.kernel_config
                    mamba_ssm.ops.kernel_config._config_cache = None
                    importlib.reload(mamba_ssm.ops.kernel_config)

                    import mamba_ssm.ops.triton.ssd_chunk_scan as ssd_module
                    importlib.reload(ssd_module)
                    
                    # Correctness check
                    out_triton, out_x_triton = ssd_module._chunk_scan_fwd(
                        cb_ref, x_ref, dt_ref, dA_cumsum_ref, C_ref, states_ref, D_t, z_t, seq_t
                    )
                    
                    # Benchmark Triton
                    triton_time = benchmark_kernel(
                        ssd_module._chunk_scan_fwd,
                        cb_ref, x_ref, dt_ref, dA_cumsum_ref, C_ref, states_ref, D_t, z_t, seq_t
                    )

                    # Run CUDA kernel
                    config["ssd_chunk_scan"]["use_cuda"] = True
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)

                    mamba_ssm.ops.kernel_config._config_cache = None
                    importlib.reload(mamba_ssm.ops.kernel_config)
                    importlib.reload(ssd_module)
                    
                    # Correctness check
                    out_cuda, out_x_cuda = ssd_module._chunk_scan_fwd(
                        cb, x, dt, dA_cumsum, C, states, D_t, z_t, seq_t
                    )
                    
                    # Benchmark CUDA
                    cuda_time = benchmark_kernel(
                        ssd_module._chunk_scan_fwd,
                        cb, x, dt, dA_cumsum, C, states, D_t, z_t, seq_t
                    )

                    rtol = 1e-2  # FP16 tolerance
                    atol = 1e-2  # FP16 tolerance

                    out_match = torch.allclose(out_triton, out_cuda, rtol=rtol, atol=atol)
                    out_x_match = (out_x_triton is None and out_x_cuda is None) or \
                                  (out_x_triton is not None and out_x_cuda is not None and
                                   torch.allclose(out_x_triton, out_x_cuda, rtol=rtol, atol=atol))

                    case_desc = f"D={use_D}, z={use_z}, seq_idx={use_seq_idx}"
                    speedup = triton_time / cuda_time if cuda_time > 0 else 0.0
                    
                    latency_info = f"  Triton: {triton_time:.3f}ms, CUDA: {cuda_time:.3f}ms, Speedup: {speedup:.2f}x"
                    
                    if out_match and out_x_match:
                        passed += 1
                        print(f"✓ batch={batch}, seqlen={seqlen}, nheads={nheads}, hdim={hdim}, dtype={dtype}, {case_desc}")
                        print(latency_info)
                    else:
                        max_diff = (out_triton - out_cuda).abs().max().item()
                        print(f"✗ batch={batch}, seqlen={seqlen}, nheads={nheads}, hdim={hdim}, dtype={dtype}, {case_desc}, max_diff={max_diff:.6e}")
                        print(latency_info)

                # Run a small set of representative flag combinations
                run_case(use_D=True, use_z=True, use_seq_idx=True)
                run_case(use_D=True, use_z=False, use_seq_idx=True)
                run_case(use_D=False, use_z=False, use_seq_idx=False)
            except Exception as e:
                print(f"✗ batch={batch}, seqlen={seqlen}, nheads={nheads}, hdim={hdim}, dtype={dtype}, error: {e}")

    print(f"\nPassed {passed}/{total} tests")
    return passed == total


def test_chunk_scan_original():
    """Original testing harness retained for quick reference."""
    from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd

    print("Testing chunk_scan_fwd kernel (original harness)...")
    device = "cuda"

    configs = [
        (1, 64, 4, 32, 1, 16, 64),
        (2, 128, 4, 64, 2, 16, 64),
        (1, 256, 8, 64, 1, 32, 64),
        (2, 256, 8, 64, 2, 32, 128),
    ]
    dtypes = [torch.float16]  # Only FP16 - CUDA kernel is WMMA-only

    passed = 0
    total = 0

    for batch, seqlen, nheads, hdim, ngroups, dstate, chunk_size in configs:
        for dtype in dtypes:
            total += 1
            try:
                nchunks = (seqlen + chunk_size - 1) // chunk_size

                cb = torch.randn(batch, nchunks, ngroups, chunk_size, chunk_size,
                                 device=device, dtype=dtype) * 0.1
                x = torch.randn(batch, seqlen, nheads, hdim, device=device, dtype=dtype)
                dt = torch.rand(batch, nheads, nchunks, chunk_size, device=device, dtype=dtype) * 0.1 + 0.01
                dA_cumsum = torch.randn(batch, nheads, nchunks, chunk_size, device=device, dtype=torch.float32) * 0.5
                C = torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=dtype) * 0.1
                states = torch.randn(batch, nchunks, nheads, hdim, dstate, device=device, dtype=dtype) * 0.1
                D = torch.randn(nheads, hdim, device=device, dtype=dtype) * 0.1
                z = torch.randn(batch, seqlen, nheads, hdim, device=device, dtype=dtype)

                cb_ref = cb.clone()
                x_ref = x.clone()
                dt_ref = dt.clone()
                dA_cumsum_ref = dA_cumsum.clone()
                C_ref = C.clone()
                states_ref = states.clone()
                D_ref = D.clone()
                z_ref = z.clone()

                config_path = Path(__file__).parent.parent / "kernels_config.json"
                with open(config_path) as f:
                    config = json.load(f)

                config["ssd_chunk_scan"]["use_cuda"] = False
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)

                import importlib
                import mamba_ssm.ops.kernel_config
                mamba_ssm.ops.kernel_config._config_cache = None
                importlib.reload(mamba_ssm.ops.kernel_config)

                import mamba_ssm.ops.triton.ssd_chunk_scan as ssd_module
                importlib.reload(ssd_module)
                out_triton, out_x_triton = ssd_module._chunk_scan_fwd(
                    cb_ref, x_ref, dt_ref, dA_cumsum_ref, C_ref, states_ref, D_ref, z_ref, None
                )

                config["ssd_chunk_scan"]["use_cuda"] = True
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)

                mamba_ssm.ops.kernel_config._config_cache = None
                importlib.reload(mamba_ssm.ops.kernel_config)
                importlib.reload(ssd_module)
                out_cuda, out_x_cuda = ssd_module._chunk_scan_fwd(
                    cb, x, dt, dA_cumsum, C, states, D, z, None
                )

                rtol = 1e-2  # FP16 tolerance
                atol = 1e-2  # FP16 tolerance

                out_match = torch.allclose(out_triton, out_cuda, rtol=rtol, atol=atol)
                out_x_match = out_x_triton is None and out_x_cuda is None or \
                              torch.allclose(out_x_triton, out_x_cuda, rtol=rtol, atol=atol)

                if out_match and out_x_match:
                    passed += 1
                    print(f"✓ batch={batch}, seqlen={seqlen}, nheads={nheads}, hdim={hdim}, dtype={dtype}")
                else:
                    max_diff = (out_triton - out_cuda).abs().max().item()
                    print(f"✗ batch={batch}, seqlen={seqlen}, nheads={nheads}, hdim={hdim}, dtype={dtype}, max_diff={max_diff:.6e}")

            except Exception as e:
                print(f"✗ batch={batch}, seqlen={seqlen}, nheads={nheads}, hdim={hdim}, dtype={dtype}, error: {e}")

    print(f"\nPassed {passed}/{total} tests (original harness)")
    return passed == total


if __name__ == "__main__":
    success = test_chunk_scan() if USE_ENHANCED_TEST else test_chunk_scan_original()
    sys.exit(0 if success else 1)
