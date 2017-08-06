"""
    Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs
    https://github.com/mys007/ecc
    https://arxiv.org/abs/1704.02901
    2017 Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import torch
import cupy.cuda
from pynvrtc.compiler import Program
from collections import namedtuple
import numpy as np

CUDA_NUM_THREADS = 1024

def GET_BLOCKS(N):
  return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS;
  
modules = {}

def get_dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'
   
def get_kernel_func(kname, ksrc, dtype):
    if kname+dtype not in modules:
        ksrc = ksrc.replace('DTYPE', dtype)
        prog = Program(ksrc.encode('utf-8'), (kname+dtype+'.cu').encode('utf-8'))
        ptx = prog.compile()
        log = prog._interface.nvrtcGetProgramLog(prog._program)
        if len(log.strip()) > 0: print(log)
        module = cupy.cuda.function.Module()
        module.load(bytes(ptx.encode()))
        modules[kname+dtype] = module
    else:
        module = modules[kname+dtype]
        
    Stream = namedtuple('Stream', ['ptr'])
    s = Stream(ptr=torch.cuda.current_stream().cuda_stream)        
        
    return module.get_function(kname), s
        
####       
       
def conv_aggregate_fw_kernel(**kwargs):
    kernel = r'''
extern "C"
__global__ void conv_aggregate_fw_kernel(DTYPE* dest, const DTYPE* src, const long long* lengths, int width, int N, int dest_stridex, int src_stridex) {
	
    int x = blockIdx.x * blockDim.x + threadIdx.x; //one thread per feature channel, runs over all nodes
    if (x >= width) return;

	for (int i=0; i<N; ++i) {	
		if (lengths[i] > 0) {
			int src_step = lengths[i] * src_stridex;
			DTYPE sum = 0;
		
			for (int j = x; j < src_step; j += src_stridex) {
				sum += src[j];
			}

			dest[x] = sum / lengths[i];			
			src += src_step;
		}
		else {
			dest[x] = 0;
		}
		
		dest += dest_stridex;
	}
}
'''
    return kernel


def conv_aggregate_bw_kernel(**kwargs):
    kernel = r'''
extern "C"
__global__ void conv_aggregate_bw_kernel(DTYPE* dest, const DTYPE* src, const long long* lengths, int width, int N, int dest_stridex, int src_stridex) {
	
    int x = blockIdx.x * blockDim.x + threadIdx.x; //one thread per feature channel, runs over all nodes
    if (x >= width) return;
	
	for (int i=0; i<N; ++i) {
		if (lengths[i] > 0) {
			int dest_step = lengths[i] * dest_stridex;	
			DTYPE val = src[x] / lengths[i];	
			
			for (int j = x; j < dest_step; j += dest_stridex) {
				dest[j] = val;
			}

			dest += dest_step;
		}
		
		src += src_stridex;
	}
}
'''
    return kernel

def conv_aggregate_fw(dest, src, degs):   
    n = degs.numel()
    w = src.size(1)
    assert n == dest.size(0) and w == dest.size(1)
    assert type(src)==type(dest) and isinstance(degs, torch.cuda.LongTensor)
    
    function, stream = get_kernel_func('conv_aggregate_fw_kernel', conv_aggregate_fw_kernel(), get_dtype(src))
    function(args=[dest.data_ptr(), src.data_ptr(), degs.data_ptr(), np.int32(w), np.int32(n), np.int32(dest.stride(0)), np.int32(src.stride(0))],
             block=(CUDA_NUM_THREADS,1,1), grid=(GET_BLOCKS(w),1,1), stream=stream)
             
                                         
def conv_aggregate_bw(dest, src, degs):
    n = degs.numel()
    w = src.size(1)
    assert n == src.size(0) and w == dest.size(1)
    assert type(src)==type(dest) and isinstance(degs, torch.cuda.LongTensor)
    
    function, stream = get_kernel_func('conv_aggregate_bw_kernel', conv_aggregate_bw_kernel(), get_dtype(src))
    function(args=[dest.data_ptr(), src.data_ptr(), degs.data_ptr(), np.int32(w), np.int32(n), np.int32(dest.stride(0)), np.int32(src.stride(0))],
             block=(CUDA_NUM_THREADS,1,1), grid=(GET_BLOCKS(w),1,1), stream=stream)                           

                                         


def maxpool_fw_kernel(**kwargs):
    kernel = r'''
extern "C"                                         
__global__ void maxpool_fw_kernel(DTYPE* dest, long long* indices, const DTYPE* src, const long long* lengths, int width, int N, int dest_stridex, int src_stridex) {
	
    int x = blockIdx.x * blockDim.x + threadIdx.x; //one thread per feature channel, runs over all points
    if (x >= width) return;
	
	for (int i=0; i<N; ++i) {		
		if (lengths[i] > 0) {
			long long src_step = lengths[i] * src_stridex;
			long long bestjj = -1;
			DTYPE best = -1e10;
			
			for (long long j = x, jj=0; j < src_step; j += src_stridex, ++jj) {
				if (src[j] > best) {
					best = src[j];
					bestjj = jj;
				}
			}
			
			dest[x] = best;
			indices[x] = bestjj;
			
			src += src_step;
		}
		else {
			dest[x] = 0;
			indices[x] = -1;
		}
		
		dest += dest_stridex;
		indices += dest_stridex;
	}
}
'''
    return kernel
    
def maxpool_bw_kernel(**kwargs):
    kernel = r'''
//also directly scatters results by dest_indices (saves one sparse intermediate buffer)
extern "C"          
__global__ void maxpool_bw_kernel(DTYPE* dest, const long long* dest_indices, const long long* max_indices, const DTYPE* src, const long long* lengths, int width, int N, int dest_stridex, int src_stridex) {
	
    int x = blockIdx.x * blockDim.x + threadIdx.x; //one thread per feature channel, runs over all points
    if (x >= width) return;
	
	for (int i=0; i<N; ++i) {
		if (lengths[i] > 0) {

            long long destidx = dest_indices[max_indices[x]];
			dest[x + destidx * dest_stridex] += src[x]; //no need for atomicadd, only one threads cares about each feat
			
			dest_indices += lengths[i];
		}
		
		src += src_stridex;
		max_indices += src_stridex;
	}
}
'''
    return kernel
    
    
def maxpool_fw(dest, indices, src, degs):   
    n = degs.numel()
    w = src.size(1)
    assert n == dest.size(0) and w == dest.size(1)
    assert type(src)==type(dest) and isinstance(degs, torch.cuda.LongTensor) and isinstance(indices, torch.cuda.LongTensor)
    
    function, stream = get_kernel_func('maxpool_fw_kernel', maxpool_fw_kernel(), get_dtype(src))
    function(args=[dest.data_ptr(), indices.data_ptr(), src.data_ptr(), degs.data_ptr(), np.int32(w), np.int32(n), np.int32(dest.stride(0)), np.int32(src.stride(0))],
             block=(CUDA_NUM_THREADS,1,1), grid=(GET_BLOCKS(w),1,1), stream=stream)    
    
def maxpool_bw(dest, idxn, indices, src, degs):   
    n = degs.numel()
    w = src.size(1)
    assert n == src.size(0) and w == dest.size(1)
    assert type(src)==type(dest) and isinstance(degs, torch.cuda.LongTensor) and isinstance(indices, torch.cuda.LongTensor) and isinstance(idxn, torch.cuda.LongTensor)
    
    function, stream = get_kernel_func('maxpool_bw_kernel', maxpool_bw_kernel(), get_dtype(src))
    function(args=[dest.data_ptr(), idxn.data_ptr(), indices.data_ptr(), src.data_ptr(), degs.data_ptr(), np.int32(w), np.int32(n), np.int32(dest.stride(0)), np.int32(src.stride(0))],
             block=(CUDA_NUM_THREADS,1,1), grid=(GET_BLOCKS(w),1,1), stream=stream)    

    

def avgpool_bw_kernel(**kwargs):
    kernel = r'''
//also directly scatters results by dest_indices (saves one intermediate buffer)
extern "C"     
__global__ void avgpool_bw_kernel(DTYPE* dest, const long long* dest_indices, const DTYPE* src, const long long* lengths, int width, int N, int dest_stridex, int src_stridex) {
	
    int x = blockIdx.x * blockDim.x + threadIdx.x; //one thread per feature channel, runs over all points
    if (x >= width) return;
	
	for (int i=0; i<N; ++i) {
		if (lengths[i] > 0) {
		
			DTYPE val = src[x] / lengths[i];
			
			for (int j = 0; j < lengths[i]; ++j) {
				long long destidx = dest_indices[j];
				dest[x + destidx * dest_stridex] += val; //no need for atomicadd, only one threads cares about each feat
			}

			dest_indices += lengths[i];
		}
		
		src += src_stridex;
	}
}
'''
    return kernel


def avgpool_fw(dest, src, degs):   
    conv_aggregate_fw(dest, src, degs)

def avgpool_bw(dest, idxn, src, degs):   
    n = degs.numel()
    w = src.size(1)
    assert n == src.size(0) and w == dest.size(1)
    assert type(src)==type(dest) and isinstance(degs, torch.cuda.LongTensor) and isinstance(idxn, torch.cuda.LongTensor)
    
    function, stream = get_kernel_func('avgpool_bw_kernel', avgpool_bw_kernel(), get_dtype(src))
    function(args=[dest.data_ptr(), idxn.data_ptr(), src.data_ptr(), degs.data_ptr(), np.int32(w), np.int32(n), np.int32(dest.stride(0)), np.int32(src.stride(0))],
             block=(CUDA_NUM_THREADS,1,1), grid=(GET_BLOCKS(w),1,1), stream=stream)      
