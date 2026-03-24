# ------------------------------
# utils/vocab_utils.py
# ------------------------------
import json

def build_vocabulary(token_sequences):
    vocab = {"PAD": 0, "START": 1, "END": 2}
    idx = 3
    for seq in token_sequences:
        for token in seq:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab

def tokens_to_ids(tokens, vocab):
    return [vocab[token] for token in tokens if token in vocab]

def ids_to_tokens(ids, inv_vocab):
    return [inv_vocab[i] for i in ids if i in inv_vocab]