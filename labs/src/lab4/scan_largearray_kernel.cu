#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 256

// Resolve bank conflict by padding
#ifdef RESOLVE_BANK_CONFLICTS
#define FREE_BANK_CONFLICT(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#else
#define FREE_BANK_CONFLICT(index) ((index) >> LOG_NUM_BANKS)
#endif

// Global vars 
float** Global_Blocks;
unsigned int Global_Mem_Alloc = 0;

// Resolve power of 2 issue 
template <bool storeSum, bool NotPow2_flag>
__global__ void PrefixSum_Kernel(float *g_out, const float *g_in, float *g_blockSums, int n, int blockIndex, int baseIndex);
__global__ void Global_Add(float *g_data, float *block_num, int n, int blockOffset, int baseIndex);

// Lab4: Host Helper Functions (allocate your own data structure...)
void MemAllocate(unsigned int num_elements)
{
    unsigned int elements = num_elements;
    int iteration = 0;
    while(elements > 1){
      int blcknum = (int)ceil((float)elements / (2.f * BLOCK_SIZE));    
      unsigned int numBlocks = 0;
      if (1 > blcknum){
        numBlocks = 1; 
      }
      else {
        numBlocks = blcknum;
      }
      if (numBlocks == blcknum)
        iteration++;
      elements = numBlocks;    
    }

    Global_Blocks = (float**) malloc(iteration * sizeof(float*));
    Global_Mem_Alloc = iteration;
    elements = num_elements;
    iteration = 0;
    
    while(elements > 1){
      int blcknum = (int)ceil((float)elements / (2.f * BLOCK_SIZE));   
      unsigned int numBlocks = 0;
      if (1 > blcknum){
        numBlocks = 1;
      }
      else{
        numBlocks = blcknum;
      }
      if (numBlocks == blcknum)
          cudaMalloc((void**) &Global_Blocks[iteration++], numBlocks * sizeof(float));
      elements = numBlocks;
    }
}

void PrefixSum(float *outArray, const float *inArray, int numElements, int iteration)
{
    int blcknum = (int)ceil((float)numElements / (2.f * BLOCK_SIZE));
    unsigned int numBlocks = 0;
    
    if (1 > blcknum){
      numBlocks = 1;
    }
    else{
      numBlocks = blcknum;
    }
    
    unsigned int numThreads;

    if (numBlocks == blcknum)
        numThreads = BLOCK_SIZE;
    else if ((numElements&(numElements-1))==0)
        numThreads = numElements / 2;
    else{
        int exp;
        frexp((float)numElements, &exp);
        numThreads = 1 << (exp - 1);
    }

    unsigned int block_elements = numThreads * 2;
    unsigned int Lblock_elements = numElements - (numBlocks - 1) * block_elements;
    unsigned int Lblock_threads = 0;
    
    if(1 > Lblock_elements / 2){
      Lblock_threads = 1;
    }
    else{
      Lblock_threads = Lblock_elements / 2;
    }
    unsigned int LBlock_nPow2_flag = 0;
    unsigned int Lblock_smem = 0;
    
    if (Lblock_elements != block_elements)
    {
        LBlock_nPow2_flag = 1;
        if((Lblock_elements&(Lblock_elements-1)) != 0){ // Check if last block is power of 2
            int exp;
            frexp((float)Lblock_elements, &exp);
            Lblock_threads = 1 << (exp - 1);
        }
        unsigned int pad = (2 * Lblock_threads) / NUM_BANKS; // padding for bank conflicts
        Lblock_smem = sizeof(float) * (2 * Lblock_threads + pad);
    }

    unsigned int pad = block_elements / NUM_BANKS;
    unsigned int sharedMemSize = sizeof(float) * (block_elements + pad);
    unsigned int gridsize = 0;

    if(1 > (numBlocks - LBlock_nPow2_flag)){
      gridsize = 1;
    }
    else{ 
      gridsize = numBlocks - LBlock_nPow2_flag;
    }
    
    dim3 Grid_Size(gridsize);
    dim3 Block_Size(numThreads);

    if (numBlocks > 1){
      PrefixSum_Kernel<true, false><<< Grid_Size, Block_Size, sharedMemSize >>>(outArray, inArray, Global_Blocks[iteration], numThreads * 2, 0, 0);
      PrefixSum(Global_Blocks[iteration], Global_Blocks[iteration], numBlocks, iteration+1);
      Global_Add<<< Grid_Size, Block_Size >>>(outArray, Global_Blocks[iteration], numElements - Lblock_elements, 0, 0);      
    }
    else if((numBlocks > 1) && LBlock_nPow2_flag){
      PrefixSum_Kernel<true, true><<< 1, Lblock_threads, Lblock_smem >>> (outArray, inArray, Global_Blocks[iteration], Lblock_elements, numBlocks - 1, numElements - Lblock_elements);
      PrefixSum(Global_Blocks[iteration], Global_Blocks[iteration], numBlocks, iteration+1);      
      Global_Add<<< 1, Lblock_threads >>>(outArray, Global_Blocks[iteration], Lblock_elements, numBlocks - 1, numElements - Lblock_elements);
    }
    else if ((numElements & (numElements-1)) == 0){
      PrefixSum_Kernel<false, false><<< Grid_Size, Block_Size, sharedMemSize >>>(outArray, inArray, 0, numThreads * 2, 0, 0);
    }
    else{
      PrefixSum_Kernel<false, true><<< Grid_Size, Block_Size, sharedMemSize >>>(outArray, inArray, 0, numElements, 0, 0);
    }
}

void MemDeallocate()
{
    for (unsigned int i = 0; i < Global_Mem_Alloc; i++)
      cudaFree(Global_Blocks[i]);
    free((void**)Global_Blocks);
    Global_Blocks = 0;
    Global_Mem_Alloc = 0;
}

// Lab4: Device Functions
template <bool storeSum>
__device__ void PrefixSumBlock(float *data, int blockIndex, float *blockSums)
{
    // Up-sweep 
    unsigned int stride = 1;
    unsigned int blckIdx = 0;     
    for (int d = blockDim.x; d > 0; d >>= 1)
    {
        __syncthreads();

        if (threadIdx.x < d){
            int i  = __mul24(__mul24(2, stride), threadIdx.x);
            int indexA = i + stride - 1;
            int indexB = indexA + stride;
            indexA += FREE_BANK_CONFLICT(indexA);
            indexB += FREE_BANK_CONFLICT(indexB);
            data[indexB] += data[indexA];
        }
        stride *= 2;
    }
    
    if (blockIndex == 0){
      blckIdx = blockIdx.x;
    }
    else {
      blckIdx = blockIndex;
    }
       
    // Down-sweep 
    if (threadIdx.x == 0)
    {
        int index = (blockDim.x << 1) - 1;
        index += FREE_BANK_CONFLICT(index);
        if (storeSum)
          blockSums[blckIdx] = data[index];
        data[index] = 0;
    }

    for (int d = 1; d <= blockDim.x; d *= 2)
    {
        stride = stride >> 1;
        __syncthreads();

        if (threadIdx.x < d)
        {
            int i  = __mul24(__mul24(2, stride), threadIdx.x);
            int indexA = i + stride - 1;
            int indexB = indexA + stride;
            indexA += FREE_BANK_CONFLICT(indexA);
            indexB += FREE_BANK_CONFLICT(indexB);
            float t  = data[indexA];
            data[indexA] = data[indexB];
            data[indexB] += t;
        }
    } 
}

// Lab4: Kernel Functions
template <bool storeSum, bool NotPow2_flag>
__global__ void PrefixSum_Kernel(float *g_out, const float *g_in, float *g_blockSums, int n, int blockIndex, int baseIndex)
{
    int indexA, indexB, m_indexA, m_indexB, bankOffsetA, bankOffsetB;

    // Load data into shared memory
    extern __shared__ float s_data[];
    unsigned int bseIdx = 0; 
    if(baseIndex == 0) {
      bseIdx = mul24(blockIdx.x, (blockDim.x << 1));
    }
    else {
      bseIdx = baseIndex; 
    }

    indexA = threadIdx.x;
    indexB = threadIdx.x + blockDim.x;
    m_indexA = bseIdx + threadIdx.x;
    m_indexB = m_indexA + blockDim.x;
    bankOffsetA = FREE_BANK_CONFLICT(indexA);
    bankOffsetB = FREE_BANK_CONFLICT(indexB);

    s_data[indexA + bankOffsetA] = g_in[m_indexA]; 
    
    // Power of 2 check
    if (NotPow2_flag){
      s_data[indexB + bankOffsetB] = (indexB < n) ? g_in[m_indexB] : 0; 
    }
    else
    {
      s_data[indexB + bankOffsetB] = g_in[m_indexB]; 
    }
    
    // Blockwise prefix-scan
    PrefixSumBlock<storeSum>(s_data, blockIndex, g_blockSums); 
    
    __syncthreads();
    
    // Global Mem data
    g_out[m_indexA] = s_data[indexA + bankOffsetA]; 
    if (NotPow2_flag && (indexB < n)){
      g_out[m_indexB] = s_data[indexB + bankOffsetB]; 
    }
    else{
      g_out[m_indexB] = s_data[indexB + bankOffsetB]; 
    }                              
}

// Global Add for every block
__global__ void Global_Add(float *g_data, float *block_num, int n, int blockOffset, int baseIndex)
{
    __shared__ float sharedId;
    if (threadIdx.x == 0)
      sharedId = block_num[blockIdx.x + blockOffset];
    
    unsigned int addr = __mul24(blockIdx.x, (blockDim.x << 1)) + baseIndex + threadIdx.x; 

    __syncthreads();
    
    g_data[addr] += sharedId;
    g_data[addr + blockDim.x] += (threadIdx.x + blockDim.x < n) * sharedId;
}

// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, int numElements)
{
	MemAllocate(numElements);
	PrefixSum(outArray, inArray, numElements, 0);
	MemDeallocate();
}
// **===-----------------------------------------------------------===**

#endif // _PRESCAN_CU_