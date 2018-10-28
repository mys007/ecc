#include <torch/torch.h>

// CUDA kernel wrappers declarations
void conv_aggregate_fw_cuda(
		at::Tensor dest,
		at::Tensor src,
		at::Tensor degs,
		at::Tensor csdegs,
		int width,
		int N,
		int dest_stridex,
		int src_stridex,
		int blockDimy
		);

void conv_aggregate_bw_cuda(
		at::Tensor dest,
		at::Tensor src,
		at::Tensor degs,
		at::Tensor csdegs,
		int width,
		int N,
		int dest_stridex,
		int src_stridex,
		int blockDimy
		);


// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.

void conv_aggregate_fw(
		at::Tensor dest,
		at::Tensor src,
		at::Tensor degs,
		at::Tensor csdegs,
		int width,
		int N,
		int dest_stridex,
		int src_stridex,
		int blockDimy
		){

	conv_aggregate_fw_cuda(dest, src, degs, csdegs, width, N, dest_stridex, src_stridex, blockDimy);

}

void conv_aggregate_bw(
		at::Tensor dest,
		at::Tensor src,
		at::Tensor degs,
		at::Tensor csdegs,
		int width,
		int N,
		int dest_stridex,
		int src_stridex,
		int blockDimy
		){

	conv_aggregate_bw_cuda(dest, src, degs, csdegs, width, N, dest_stridex, src_stridex, blockDimy);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("conv_aggregate_fw", &conv_aggregate_fw_cuda, "conv_aggregate_fw");
	m.def("conv_aggregate_bw", &conv_aggregate_bw_cuda, "conv_aggregate_bw");
}