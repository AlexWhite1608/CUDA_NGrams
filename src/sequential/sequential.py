from collections import Counter
from typing import Dict
from collections import Counter
from typing import Dict, List, Tuple

# Compute character N-grams using a sequential approach. Returns a dictionary with N-gram counts.
def compute_char_ngrams_cpu(text: str, n: int) -> Dict[str, int]:
    
    if len(text) < n:
        return {}
    
    ngrams = []
    for i in range(len(text) - n + 1):
        ngram = text[i:i+n]
        ngrams.append(ngram)
    
    return dict(Counter(ngrams))