# ------------------------------
# training/trainer.py
# ------------------------------
from tqdm import tqdm
import torch

def train(model, dataloader, optimizer, criterion, vocab_size, device):
    model.train()
    total_loss = 0
    total_tokens = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training')
    for i, batch in progress_bar:
        batch = batch.to(device)
        input_seq = batch[:, :-1]
        target_seq = batch[:, 1:]
        output = model(input_seq)
        loss = criterion(
            output.view(-1, output.size(-1)),
            target_seq.reshape(-1)
        )
        # loss = criterion(output.view(-1, vocab_size), target_seq.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # contar tokens válidos (no padding)
        valid_tokens = (target_seq != 0).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens
        progress_bar.set_postfix(loss=loss.item())
    return total_loss / total_tokens
    #     total_loss += loss.item()
    #     progress_bar.set_postfix(loss=loss.item())
    # return total_loss / len(dataloader)