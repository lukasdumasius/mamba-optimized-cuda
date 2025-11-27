#!/usr/bin/env python3
import torch
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_swiglu():
    from mamba_ssm.ops.triton.k_activations import _swiglu_fwd, _swiglu_bwd
    
    print("Testing SwiGLU kernel...")
    device = "cuda"
    batch_sizes = [1, 2, 4, 8]
    dims = [64, 128, 256, 512, 1024]
    dtypes = [torch.float32, torch.float16]
    
    passed = 0
    total = 0
    
    for batch in batch_sizes:
        for dim in dims:
            for dtype in dtypes:
                total += 1
                try:
                    xy = torch.randn(batch, dim * 2, device=device, dtype=dtype, requires_grad=True)
                    xy_ref = xy.clone().detach().requires_grad_(True)
                    
                    config_path = Path(__file__).parent.parent / "kernels_config.json"
                    with open(config_path) as f:
                        config = json.load(f)
                    
                    config["swiglu"]["use_cuda"] = False
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    
                    import importlib
                    import mamba_ssm.ops.kernel_config
                    importlib.reload(mamba_ssm.ops.kernel_config)
                    
                    from mamba_ssm.ops.triton.k_activations import _swiglu_fwd as _swiglu_fwd_triton
                    out_triton = _swiglu_fwd_triton(xy_ref)
                    
                    config["swiglu"]["use_cuda"] = True
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    
                    importlib.reload(mamba_ssm.ops.kernel_config)
                    from mamba_ssm.ops.triton.k_activations import _swiglu_fwd as _swiglu_fwd_cuda
                    out_cuda = _swiglu_fwd_cuda(xy)
                    
                    rtol = 1e-3 if dtype == torch.float16 else 1e-5
                    atol = 1e-3 if dtype == torch.float16 else 1e-6
                    
                    if torch.allclose(out_triton, out_cuda, rtol=rtol, atol=atol):
                        passed += 1
                        print(f"✓ batch={batch}, dim={dim}, dtype={dtype}")
                    else:
                        max_diff = (out_triton - out_cuda).abs().max().item()
                        print(f"✗ batch={batch}, dim={dim}, dtype={dtype}, max_diff={max_diff:.6e}")
                        
                except Exception as e:
                    print(f"✗ batch={batch}, dim={dim}, dtype={dtype}, error: {e}")
    
    print(f"\nPassed {passed}/{total} tests")
    return passed == total

if __name__ == "__main__":
    success = test_swiglu()
    sys.exit(0 if success else 1)

