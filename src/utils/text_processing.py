import re
import numpy as np
from typing import Tuple, Dict, List

# Reads a text file and amplifies its content by concatenating it multiple times
def amplify_corpus(filepath: str, amplification_factor: int) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    amplified = text * amplification_factor
    size_mb = len(amplified.encode('utf-8')) / (1024 * 1024)
    print(f"Amplified corpus: {size_mb:.2f} MB")
    
    return amplified

# Pre-elaborates text: lowercases, removes punctuation and digits, splits into tokens
def tokenize(text: str) -> List[str]:
    text = text.lower()
    
    # removes punctuation, replacing with space
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # removes digits
    text = re.sub(r'\d+', '', text)
    
    # splits into tokens based on whitespace
    tokens = text.split()
    tokens = [t for t in tokens if t.strip()]
    
    print(f"Token number: {len(tokens)}")
    
    return tokens

# Creates a bidirectional mapping between words and integer IDs (word_to_id, id_to_word)
# transforms each word into a unique integer ID to facilitate GPU processing
def create_vocabulary(tokens: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    
    unique_words = sorted(set(tokens))
    word_to_id = {word: idx for idx, word in enumerate(unique_words)}
    id_to_word = {idx: word for word, idx in word_to_id.items()}
    
    print(f"Vocabolary dimension: {len(word_to_id)}")
    
    return word_to_id, id_to_word

# Converts a list of tokens into a numpy array of integer IDs
def tokens_to_ids(tokens: List[str], word_to_id: Dict[str, int]) -> np.ndarray:
    
    ids = np.array([word_to_id[token] for token in tokens], dtype=np.uint32)
    return ids

# Converts a text string into a numpy array of bytes
def text_to_bytes(text: str) -> np.ndarray:
    
    return np.frombuffer(text.encode('latin-1', errors='ignore'), dtype=np.uint8)