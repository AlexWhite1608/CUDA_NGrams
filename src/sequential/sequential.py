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

# Compute word N-grams using a sequential approach. Returns a dictionary with N-gram counts.
def compute_word_ngrams_cpu(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:

    if len(tokens) < n:
        return {}
    
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    
    return dict(Counter(ngrams))