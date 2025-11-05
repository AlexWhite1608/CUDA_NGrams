extern "C" {

// A-V1 kernel to compute character N-grams and update histogram with atomic operations
__global__ void char_ngram_kernel(
    const unsigned char* text,
    unsigned int text_length,
    unsigned int n,
    unsigned int* histogram,
    unsigned int hist_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // check for bounds
    if (idx >= text_length - n + 1) {
        return;
    }
    
    // computes the index
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

// A-V2 kernel 1
__global__ void char_ngram_kernel_private(
    const unsigned char* text,
    unsigned int text_length,
    unsigned int n,
    unsigned int* private_histograms,  // Array: [num_private_hists * hist_size]
    unsigned int hist_size,
    unsigned int num_private_hists     
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // check for bounds
    if (idx >= text_length - n + 1) {
        return;
    }
    
    // computes the n-gram index
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

// A-V2 kernel 2
__global__ void reduce_histograms(
    const unsigned int* private_histograms,  // Input: [num_blocks * hist_size]
    unsigned int* global_histogram,          // Output: [hist_size]
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

} // extern "C"