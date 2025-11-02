extern "C" {

// character kernel
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
        multiplier *= 256;
    }
    
    if (flat_idx < hist_size) {
        atomicAdd(&histogram[flat_idx], 1);
    }
}


// words kernel
__global__ void word_ngram_map_kernel(
    const unsigned int* word_ids,
    unsigned int num_tokens,
    unsigned int n,
    unsigned int vocab_size,
    unsigned long long* ngram_ids
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // check for bounds
    if (idx >= num_tokens - n + 1) {
        return;
    }
    
    // computes ngram id
    unsigned long long ngram_id = 0;
    unsigned long long multiplier = 1;
    
    for (int i = n - 1; i >= 0; i--) {
        ngram_id += word_ids[idx + i] * multiplier;
        multiplier *= vocab_size;
    }
    
    ngram_ids[idx] = ngram_id;
}


// counts words ngram after sorting
__global__ void word_ngram_reduce_kernel(
    const unsigned long long* sorted_ngram_ids,
    unsigned int num_ngrams,
    unsigned long long* unique_ngrams,
    unsigned int* counts,
    unsigned int* num_unique
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_ngrams) {
        return;
    }
    
    bool is_new_group = (idx == 0) || (sorted_ngram_ids[idx] != sorted_ngram_ids[idx - 1]);
    
    if (is_new_group) {
        // finds the end of the group
        unsigned int count = 1;
        unsigned int j = idx + 1;
        while (j < num_ngrams && sorted_ngram_ids[j] == sorted_ngram_ids[idx]) {
            count++;
            j++;
        }
        
        unsigned int out_idx = atomicAdd(num_unique, 1);
        unique_ngrams[out_idx] = sorted_ngram_ids[idx];
        counts[out_idx] = count;
    }
}

} // extern "C"