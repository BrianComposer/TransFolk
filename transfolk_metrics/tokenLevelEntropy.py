import math
from collections import Counter

def token_entropy(sequences, log_base=2):
    """
    sequences: lista de listas de tokens (int)
    """
    # concatenar todo
    all_tokens = [tok for seq in sequences for tok in seq]
    T = len(all_tokens)
    counts = Counter(all_tokens)

    H = 0.0
    for v, c in counts.items():
        p = c / T
        if p > 0:
            H -= p * (math.log(p) / math.log(log_base))  # bits si base=2
    return H
