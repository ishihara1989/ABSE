#include <torch/extension.h>
#include <vector>

torch::Tensor soft_dtw_cuda_forward(torch::Tensor D, float gamma);
torch::Tensor soft_dtw_cuda_backward(torch::Tensor grad, torch::Tensor D, float gamma);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor soft_dtw_forward(torch::Tensor D, float gamma){
    CHECK_INPUT(D);
    TORCH_CHECK(D.sizes().size()==3, "D must have 3 dims");
    return soft_dtw_cuda_forward(D, gamma);
}

torch::Tensor soft_dtw_backward(torch::Tensor D, torch::Tensor R, float gamma){
    CHECK_INPUT(R);
    CHECK_INPUT(D);
    TORCH_CHECK(D.sizes().size()==3, "D must have 3 dims");
    TORCH_CHECK(R.sizes().size()==3, "R must have 3 dims");
    return soft_dtw_cuda_backward(D, R, gamma);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &soft_dtw_forward, "Soft DTW forward (CUDA)");
  m.def("backward", &soft_dtw_backward, "Soft DTW backward (CUDA)");
}