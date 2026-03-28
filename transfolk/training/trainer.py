# ------------------------------
# training/trainer.py
# ------------------------------
from tqdm import tqdm
import torch

def train(model, dataloader, optimizer, criterion, vocab_size, device, scheduler=None):
    model.train()
    total_loss = 0
    total_tokens = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training')
    for i, batch in progress_bar:
        batch = batch.to(device)
        input_seq = batch[:, :-1]
        target_seq = batch[:, 1:]
        output = model(input_seq)
        # soporte KV cache
        if isinstance(output, tuple):
            output = output[0]

        B, T, V = output.shape

        loss = criterion(
            output.reshape(B * T, V),
            target_seq.reshape(B * T)
        )
        # loss = criterion(
        #     output.view(-1, output.size(-1)),
        #     target_seq.reshape(-1)
        # )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 🔥 MUY recomendable en transformers
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # contar tokens válidos (no padding)
        valid_tokens = (target_seq != 0).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens
        progress_bar.set_postfix(loss=loss.item())
    return total_loss / total_tokens
    #     total_loss += loss.item()
    #     progress_bar.set_postfix(loss=loss.item())
    # return total_loss / len(dataloader)