
# ------------------------------
# model/transformer.py
# ------------------------------
import torch
import torch.nn as nn

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_linear = nn.Linear(d_model, vocab_size)

    def _generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.token_embedding(x) + self.positional_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        x = self.transformer(x, mask=mask)
        return self.output_linear(x)
