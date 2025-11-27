#include <torch/extension.h>

torch::Tensor swiglu_fwd(torch::Tensor x, torch::Tensor y);
std::vector<torch::Tensor> swiglu_bwd(torch::Tensor x, torch::Tensor y, torch::Tensor dout, bool recompute_output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_fwd", &swiglu_fwd, "SwiGLU forward (CUDA)");
    m.def("swiglu_bwd", &swiglu_bwd, "SwiGLU backward (CUDA)");
}

