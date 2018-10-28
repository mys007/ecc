#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_NUM_THREADS 1024
#define GET_BLOCKS(N) ((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS)


template <typename scalar_t>
__global__ void conv_aggregate_fw_kernel(
		scalar_t* __restrict__ dest,
		const scalar_t* __restrict__ src,
		const int64_t* __restrict__ lengths,
		const int64_t* __restrict__ cslengths,
		int width,
		int N,
		int dest_stridex,
		int src_stridex,
		int blockDimy
		) {

    int x = blockIdx.x * blockDim.x + threadIdx.x; //one thread per feature channel, runs over all nodes
    if (x >= width) return;

    int i = blockIdx.y * blockDimy;
    int imax = min(N, i + blockDimy);
    dest += dest_stridex * i + x;
    src += src_stridex * (cslengths[i] - lengths[i]) + x;

	for (; i<imax; ++i) {
        int len = lengths[i];
		if (len > 0) {
			scalar_t sum = 0;
            for (int j=0; j<len; j++, src += src_stridex) {
                sum += *src;
			}

            *dest = sum / len;
		}
		else {
			*dest = 0;
		}

		dest += dest_stridex;
	}
}


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
		){

	//TODO: all the integer parameters can be computed here, no reason to pass them as arguments

	const auto w = src.sizes()[1];
	const dim3 grid(GET_BLOCKS(w), N / blockDimy + 1, 1);
	const dim3 block(CUDA_NUM_THREADS, 1, 1);

	AT_DISPATCH_FLOATING_TYPES(dest.type(), "conv_aggregate_fw_cuda", ([&] {
				conv_aggregate_fw_kernel<scalar_t><<<grid, block>>>(
						dest.data<scalar_t>(),
						src.data<scalar_t>(),
						degs.data<int64_t>(),
						csdegs.data<int64_t>(),
						width,
						N,
						dest_stridex,
						src_stridex,
						blockDimy);
				}));
}




template <typename scalar_t>
__global__ void conv_aggregate_bw_kernel(
		scalar_t* __restrict__ dest,
		const scalar_t* __restrict__ src,
		const int64_t* __restrict__ lengths,
		const int64_t* __restrict__ cslengths,
		int width,
		int N,
		int dest_stridex,
		int src_stridex,
		int blockDimy
		) {

    int x = blockIdx.x * blockDim.x + threadIdx.x; //one thread per feature channel, runs over all nodes
    if (x >= width) return;

    int i = blockIdx.y * blockDimy;
    int imax = min(N, i + blockDimy);
    dest += dest_stridex * (cslengths[i] - lengths[i]) + x;
    src += src_stridex * i + x;

	for (; i<imax; ++i) {
        int len = lengths[i];
		if (len > 0) {
			scalar_t val = *src / len;
            for (int j=0; j<len; j++, dest += dest_stridex) {
                *dest = val;
			}
		}

		src += src_stridex;
	}
}



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
		){

	//TODO: all the integer parameters can be computed here, no reason to pass them as arguments

	const auto w = src.sizes()[1];
	const dim3 grid(GET_BLOCKS(w), N / blockDimy + 1, 1);
	const dim3 block(CUDA_NUM_THREADS, 1, 1);

	AT_DISPATCH_FLOATING_TYPES(dest.type(), "conv_aggregate_bw_cuda", ([&] {
				conv_aggregate_bw_kernel<scalar_t><<<grid, block>>>(
						dest.data<scalar_t>(),
						src.data<scalar_t>(),
						degs.data<int64_t>(),
						csdegs.data<int64_t>(),
						width,
						N,
						dest_stridex,
						src_stridex,
						blockDimy);
				}));
}











