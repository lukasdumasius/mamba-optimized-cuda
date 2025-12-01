#include <torch/extension.h>

torch::Tensor swiglu_fwd(torch::Tensor x, torch::Tensor y);
std::vector<torch::Tensor> swiglu_bwd(torch::Tensor x, torch::Tensor y, torch::Tensor dout, bool recompute_output);

std::vector<torch::Tensor> chunk_scan_fwd_cuda(
    torch::Tensor cb,
    torch::Tensor x,
    torch::Tensor dt,
    torch::Tensor dA_cumsum,
    torch::Tensor C,
    torch::Tensor prev_states,
    c10::optional<torch::Tensor> D,
    c10::optional<torch::Tensor> z,
    c10::optional<torch::Tensor> seq_idx
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_fwd", &swiglu_fwd, "SwiGLU forward (CUDA)");
    m.def("swiglu_bwd", &swiglu_bwd, "SwiGLU backward (CUDA)");
    m.def("chunk_scan_fwd", &chunk_scan_fwd_cuda, "Chunk scan forward (CUDA)",
          py::arg("cb"), py::arg("x"), py::arg("dt"), py::arg("dA_cumsum"),
          py::arg("C"), py::arg("prev_states"),
          py::arg("D") = py::none(), py::arg("z") = py::none(), py::arg("seq_idx") = py::none());
}

