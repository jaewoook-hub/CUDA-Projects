#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

__global__ void opt_stoprollover_Kernel(uint32_t *, uint8_t *);
__global__ void opt_2dhisto_Kernel(uint32_t *, size_t , size_t , uint32_t *);

void opt_2dhisto(uint32_t *input, size_t height, size_t width, uint32_t *bin_32, uint8_t *bin_8)
{
    /* This function should only contain a call to the GPU histogramming kernel. 
       Any memory allocations and transfers must be done outside this function  */
    dim3 DimGrid (width);
    dim3 DimBlock (256); // 0~255
    opt_2dhisto_Kernel<<<DimGrid, DimBlock>>>(input, height, width, bin_32);
    opt_stoprollover_Kernel<<<4, 256>>>(bin_32, bin_8); 
    cudaThreadSynchronize();
}

// Additional Functions Below

__global__ void opt_stoprollover_Kernel(uint32_t *bin_32, uint8_t *bin_8)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // if bin_32[idx] < 255, then bin_8 = 1 * bin_32[idx]
    // if bin_32[idx] >= 255, then bin_8 = 1 * 255
    bin_8[idx] = (uint8_t)((bin_32[idx] < UINT8_MAX) * bin_32[idx]) + ((bin_32[idx] >= UINT8_MAX) * UINT8_MAX);
    __syncthreads();
}

__global__ void opt_2dhisto_Kernel(uint32_t *input, size_t width, size_t height, uint32_t *bin_32)
{

    const int threadidx = threadIdx.x + blockIdx.x * blockDim.x;
    const int blockidx = gridDim.x * blockDim.x;	
    
    // make sure all values begin from 0
    if (threadidx < 1024) 
    {
        bin_32[threadidx] = 0;
    }
    
    __syncthreads();
    
    uint32_t size = width * height;
    for (unsigned int i = threadidx; i < size; i += blockidx) 
    {
      	const int value = input[i];
      	if (bin_32[value] < UINT8_MAX && value) 
        {
            // atomicAdd - keeps the count accidentally not being counted 
      		  atomicAdd(&(bin_32[value]), 1);
      	}
    }
    
    __syncthreads();
    
}

// Below could also just be used in the main function as well - copied from previous labs

void copyToDeviceMemory(void* device, void* host, size_t size) 
{
	cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
}

void copyToHostMemory(void* host, void* device, size_t size) 
{
	cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);
}

void* allocateDeviceMemory(size_t size) 
{
	void *p;
	cudaMalloc(&p, size);
	return p;
}

void freeDeviceMemory(void* device) 
{
	cudaFree(device);
}





