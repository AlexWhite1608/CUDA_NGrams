extern "C" {

// V1 kernel to compute character N-grams and update histogram with atomic operations
__global__ void char_ngram_kernel(
    const unsigned char* text,
    unsigned int text_length,
    unsigned int n,
    unsigned int* histogram,
    unsigned int hist_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= text_length - n + 1) {
        return;
    }
    
    unsigned int flat_idx = 0;
    unsigned int multiplier = 1;
    
    // builds the index from right to left
    for (int i = n - 1; i >= 0; i--) {
        flat_idx += text[idx + i] * multiplier;
        multiplier *= 256;      // computes unique index in base 256
    }
    
    if (flat_idx < hist_size) {
        atomicAdd(&histogram[flat_idx], 1);
    }
}

// V2 kernel 1
__global__ void char_ngram_kernel_private(
    const unsigned char* text,
    unsigned int text_length,
    unsigned int n,
    unsigned int* private_histograms,
    unsigned int hist_size,
    unsigned int num_private_hists     
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= text_length - n + 1) {
        return;
    }
    
    unsigned int flat_idx = 0;
    unsigned int multiplier = 1;
    
    for (int i = n - 1; i >= 0; i--) {
        flat_idx += text[idx + i] * multiplier;
        multiplier *= 256;
    }
    
    if (flat_idx < hist_size) {
        unsigned int hist_id = blockIdx.x % num_private_hists;
        unsigned int private_offset = hist_id * hist_size;
        atomicAdd(&private_histograms[private_offset + flat_idx], 1);
    }
}

// V2 kernel 2
__global__ void reduce_histograms(
    const unsigned int* private_histograms,  
    unsigned int* global_histogram,          
    unsigned int num_blocks,
    unsigned int hist_size
) {
    unsigned int bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (bin_idx >= hist_size) {
        return;
    }
    
    unsigned int sum = 0;
    
    for (unsigned int block = 0; block < num_blocks; block++) {
        sum += private_histograms[block * hist_size + bin_idx];
    }
    
    global_histogram[bin_idx] = sum;
}

// Kernel B for map operation
__global__ void char_ngram_map_kernel(
    const unsigned char* text,
    unsigned int text_length,
    unsigned int n,
    unsigned long long* ngram_ids_output 
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= text_length - n + 1) {
        return;
    }
    
    unsigned long long flat_idx = 0;
    unsigned long long multiplier = 1;
    
    for (int i = n - 1; i >= 0; i--) {
        flat_idx += text[idx + i] * multiplier;
        multiplier *= 256;
    }
    
    // write the computed n-gram ID to output array
    ngram_ids_output[idx] = flat_idx;
}

} // extern "C"