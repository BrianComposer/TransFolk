import torch
import torch.nn.functional as F
import math

@torch.no_grad()
def conditional_entropy_k(model, seq, k, device='cpu', log_base=2):
    model.eval()
    x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
    logits = model(x)
    log_probs = F.log_softmax(logits, dim=-1)

    T = x.size(1)
    fb = 1.0 / math.log(log_base)

    total = 0.0
    count = 0
    for t in range(k, T):
        lp = log_probs[0, t-1, x[0, t]].item()
        total -= lp * fb
        count += 1

    return total / max(1, count)

def compute_Hk_from_sequences(model, sequences, k, device='cpu'):
    results = []
    for seq in sequences:
        if len(seq) > 512:      # límite del modelo actual
            seq = seq[:512]
        if len(seq) > k:
            results.append(conditional_entropy_k(model, seq, k, device))
    mean_Hk = sum(results) / len(results) if results else 0.0
    return mean_Hk, results

