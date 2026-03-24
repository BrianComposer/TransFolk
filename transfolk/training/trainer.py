# ------------------------------
# training/trainer.py
# ------------------------------
from tqdm import tqdm
import torch

def train(model, dataloader, optimizer, criterion, vocab_size, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training')
    for i, batch in progress_bar:
        batch = batch.to(device)
        input_seq = batch[:, :-1]
        target_seq = batch[:, 1:]
        output = model(input_seq)
        loss = criterion(output.view(-1, vocab_size), target_seq.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)