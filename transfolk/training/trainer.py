from __future__ import annotations

from tqdm import tqdm
import torch


def train(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    scheduler=None,
    pad_token_id: int = 0,
    grad_clip: float = 1.0,
):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")

    for _, batch in progress_bar:
        batch = batch.to(device)

        input_seq = batch[:, :-1]
        target_seq = batch[:, 1:]

        attention_mask = input_seq != pad_token_id

        output = model(
            input_seq,
            attention_mask=attention_mask,
            return_dict=False,
        )

        B, T, V = output.shape

        loss = criterion(
            output.reshape(B * T, V),
            target_seq.reshape(B * T),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        valid_tokens = (target_seq != pad_token_id).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

        progress_bar.set_postfix(
            loss=loss.item(),
            lr=optimizer.param_groups[0]["lr"],
        )

    if total_tokens == 0:
        return 0.0

    return total_loss / total_tokens
