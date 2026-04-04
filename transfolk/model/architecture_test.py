import torch
from typing import Dict, Any
from tqdm import tqdm


def _extract_logits(output):
    if isinstance(output, dict):
        return output["logits"]
    return output


def test_architecture(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    vocab_size: int,
    max_batches_overfit: int = 200,
    pad_token_id: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:

    model = model.to(device)
    results = {}

    print("\n🧪 Running architecture test...\n")

    # =========================================================
    # 1. SANITY FORWARD
    # =========================================================
    try:
        print("🔹 [1/4] Sanity forward...")

        model.eval()

        x = torch.randint(1, vocab_size, (2, 16)).to(device)
        mask = (x != pad_token_id)

        with torch.no_grad():
            out = model(x, attention_mask=mask)
            out = _extract_logits(out)

        results["sanity_shape"] = tuple(out.shape)
        results["sanity_has_nan"] = out.isnan().any().item()
        results["sanity_has_inf"] = out.isinf().any().item()

    except Exception as e:
        results["sanity_error"] = str(e)
        return results

    # =========================================================
    # 2. PADDING MASK TEST
    # =========================================================
    try:
        print("🔹 [2/4] Padding mask test...")

        model.eval()

        x_clean = torch.randint(1, vocab_size, (1, 16)).to(device)
        x_pad = x_clean.clone()
        x_pad[0, 8:] = pad_token_id

        mask_clean = (x_clean != pad_token_id)
        mask_pad = (x_pad != pad_token_id)

        with torch.no_grad():
            out_clean = _extract_logits(model(x_clean, attention_mask=mask_clean))
            out_pad = _extract_logits(model(x_pad, attention_mask=mask_pad))

        diff = (out_clean[:, :8] - out_pad[:, :8]).abs().mean().item()
        results["padding_diff"] = diff

    except Exception as e:
        results["padding_error"] = str(e)

    # =========================================================
    # 3. CAUSAL MASK TEST
    # =========================================================
    try:
        print("🔹 [3/4] Causal mask test...")

        model.eval()

        x = torch.randint(1, vocab_size, (1, 16)).to(device)
        mask = (x != pad_token_id)

        with torch.no_grad():
            out1 = _extract_logits(model(x, attention_mask=mask))

        x_future = x.clone()
        x_future[0, 10:] = torch.randint(1, vocab_size, (6,)).to(device)

        with torch.no_grad():
            out2 = _extract_logits(model(x_future, attention_mask=mask))

        diff = (out1[:, :10] - out2[:, :10]).abs().mean().item()
        results["causal_diff"] = diff

    except Exception as e:
        results["causal_error"] = str(e)

    # =========================================================
    # 4. OVERFIT TEST (CRÍTICO)
    # =========================================================
    try:
        print("🔹 [4/4] Overfit test (this is the slow part)...")

        model.train()

        batch = next(iter(dataloader)).to(device)

        input_seq = batch[:, :-1]
        target_seq = batch[:, 1:]
        mask = (input_seq != pad_token_id)

        losses = []

        progress = tqdm(
            range(max_batches_overfit),
            desc="Overfit",
            leave=True,
        )

        for i in progress:
            optimizer.zero_grad(set_to_none=True)

            output = model(input_seq, attention_mask=mask)
            output = _extract_logits(output)

            B, T, V = output.shape

            loss = criterion(
                output.reshape(B * T, V),
                target_seq.reshape(B * T)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            losses.append(loss_val)

            # 🔥 actualizar barra
            progress.set_postfix(loss=f"{loss_val:.4f}")

        results["overfit_initial_loss"] = losses[0]
        results["overfit_final_loss"] = losses[-1]
        results["overfit_min_loss"] = min(losses)

    except Exception as e:
        results["overfit_error"] = str(e)

    # =========================================================
    # PRINT FINAL
    # =========================================================
    if verbose:
        print("\n===== ARCH TEST RESULTS =====")

        for k, v in results.items():
            print(f"{k}: {v}")

        print("=============================\n")

    return results

# # architecture_test.py
#
# import torch
# from typing import Dict, Any
#
#
# def _extract_logits(output):
#     """
#     Compatible con:
#     - logits directos
#     - dict {"logits": ...}
#     """
#     if isinstance(output, dict):
#         return output["logits"]
#     return output
#
#
# def test_architecture(
#     model,
#     dataloader,
#     optimizer,
#     criterion,
#     device,
#     vocab_size: int,
#     max_batches_overfit: int = 200,
#     pad_token_id: int = 0,
#     verbose: bool = True,
# ) -> Dict[str, Any]:
#
#     model = model.to(device)
#     results = {}
#
#     # =========================================================
#     # 1. SANITY FORWARD
#     # =========================================================
#     try:
#         model.eval()
#
#         x = torch.randint(1, vocab_size, (2, 16)).to(device)
#         mask = (x != pad_token_id)
#
#         with torch.no_grad():
#             out = model(x, attention_mask=mask)
#             out = _extract_logits(out)
#
#         results["sanity_shape"] = tuple(out.shape)
#         results["sanity_has_nan"] = out.isnan().any().item()
#         results["sanity_has_inf"] = out.isinf().any().item()
#
#     except Exception as e:
#         results["sanity_error"] = str(e)
#         return results
#
#     # =========================================================
#     # 2. PADDING MASK TEST
#     # =========================================================
#     try:
#         model.eval()
#
#         x_clean = torch.randint(1, vocab_size, (1, 16)).to(device)
#         x_pad = x_clean.clone()
#         x_pad[0, 8:] = pad_token_id
#
#         mask_clean = (x_clean != pad_token_id)
#         mask_pad = (x_pad != pad_token_id)
#
#         with torch.no_grad():
#             out_clean = _extract_logits(model(x_clean, attention_mask=mask_clean))
#             out_pad = _extract_logits(model(x_pad, attention_mask=mask_pad))
#
#         diff = (out_clean[:, :8] - out_pad[:, :8]).abs().mean().item()
#         results["padding_diff"] = diff
#
#     except Exception as e:
#         results["padding_error"] = str(e)
#
#     # =========================================================
#     # 3. CAUSAL MASK TEST
#     # =========================================================
#     try:
#         model.eval()
#
#         x = torch.randint(1, vocab_size, (1, 16)).to(device)
#         mask = (x != pad_token_id)
#
#         with torch.no_grad():
#             out1 = _extract_logits(model(x, attention_mask=mask))
#
#         x_future = x.clone()
#         x_future[0, 10:] = torch.randint(1, vocab_size, (6,)).to(device)
#
#         with torch.no_grad():
#             out2 = _extract_logits(model(x_future, attention_mask=mask))
#
#         diff = (out1[:, :10] - out2[:, :10]).abs().mean().item()
#         results["causal_diff"] = diff
#
#     except Exception as e:
#         results["causal_error"] = str(e)
#
#     # =========================================================
#     # 4. OVERFIT TEST (CRÍTICO)
#     # =========================================================
#     try:
#         model.train()
#
#         batch = next(iter(dataloader)).to(device)
#
#         input_seq = batch[:, :-1]
#         target_seq = batch[:, 1:]
#         mask = (input_seq != pad_token_id)
#
#         losses = []
#
#         for i in range(max_batches_overfit):
#             optimizer.zero_grad(set_to_none=True)
#
#             output = model(input_seq, attention_mask=mask)
#             output = _extract_logits(output)
#
#             B, T, V = output.shape
#
#             loss = criterion(
#                 output.reshape(B * T, V),
#                 target_seq.reshape(B * T)
#             )
#
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#
#             losses.append(loss.item())
#
#         results["overfit_initial_loss"] = losses[0]
#         results["overfit_final_loss"] = losses[-1]
#         results["overfit_min_loss"] = min(losses)
#
#     except Exception as e:
#         results["overfit_error"] = str(e)
#
#     # =========================================================
#     # PRINT
#     # =========================================================
#     if verbose:
#         print("\n===== ARCH TEST =====")
#
#         for k, v in results.items():
#             print(f"{k}: {v}")
#
#         print("=====================\n")
#
#     return results