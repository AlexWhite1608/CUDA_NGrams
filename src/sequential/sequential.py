from collections import Counter
from typing import Dict
from collections import Counter
from typing import Dict, List, Tuple
from src.utils.text_processing import text_to_bytes

# Compute character N-grams using a sequential approach. Returns a dictionary with N-gram counts.
def compute_char_ngrams_cpu(text: str, n: int) -> Dict[str, int]:
    text_bytes = text_to_bytes(text).tobytes()
    length = len(text_bytes)
    if length < n:
        return {}

    counts = Counter()
    end = length - n + 1
    for i in range(end):
        gram = text_bytes[i:i + n]            
        gram_str = gram.decode('latin-1')    
        counts[gram_str] += 1

    return dict(counts)