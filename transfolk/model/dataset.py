# ------------------------------
# model/dataset.py
# ------------------------------
import torch
from torch.utils.data import Dataset

class MusicDataset(Dataset):
    def __init__(self, sequences, max_seq_len=512, pad_token_id=0):
        self.sequences = sequences
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]
        else:
            seq = seq + [self.pad_token_id] * (self.max_seq_len - len(seq))
        return torch.tensor(seq, dtype=torch.long)
