#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint32_t*, size_t, size_t, uint32_t*, uint8_t*);
//void opt_stoprollover_Kernel(uint32_t*, uint8_t*)
//void opt_2dhisto_Kernel(uint32_t*, size_t, size_t, uint32_t*)
void* allocateDeviceMemory(size_t);
void copyToDeviceMemory(void*, void*, size_t);
void copyToHostMemory(void*, void*, size_t);
void freeDeviceMemory(void*);

#endif