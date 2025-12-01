#!/usr/bin/env python3
import torch
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_chunk_scan():
    from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd
    
    print("Testing chunk_scan_fwd kernel...")
    device = "cuda"
    
    # Test configurations: (batch, seqlen, nheads, hdim, ngroups, dstate, chunk_size)
    configs = [
        (1, 64, 4, 32, 1, 16, 64),
        (2, 128, 4, 64, 2, 16, 64),
        (1, 256, 8, 64, 1, 32, 64),
        (2, 256, 8, 64, 2, 32, 128),
    ]
    dtypes = [torch.float32, torch.float16]
    
    passed = 0
    total = 0
    
    for batch, seqlen, nheads, hdim, ngroups, dstate, chunk_size in configs:
        for dtype in dtypes:
            total += 1
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
                
                # Clone inputs for reference
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
                out_triton, out_x_triton = ssd_module._chunk_scan_fwd(
                    cb_ref, x_ref, dt_ref, dA_cumsum_ref, C_ref, states_ref, D_ref, z_ref, None
                )
                
                # Run CUDA kernel
                config["ssd_chunk_scan"]["use_cuda"] = True
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                mamba_ssm.ops.kernel_config._config_cache = None
                importlib.reload(mamba_ssm.ops.kernel_config)
                importlib.reload(ssd_module)
                out_cuda, out_x_cuda = ssd_module._chunk_scan_fwd(
                    cb, x, dt, dA_cumsum, C, states, D, z, None
                )
                
                rtol = 1e-2 if dtype == torch.float16 else 1e-4
                atol = 1e-2 if dtype == torch.float16 else 1e-5
                
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
    
    print(f"\nPassed {passed}/{total} tests")
    return passed == total

if __name__ == "__main__":
    success = test_chunk_scan()
    sys.exit(0 if success else 1)
