from __future__ import annotations
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GCGNaNGradientError(Exception):
    def __init__(self, step: int, position: int):
        super().__init__(f"NaN gradient at step={step}, position={position}")
        self.step = step
        self.position = position


def token_gradients(
    model,
    input_ids: torch.Tensor,
    target_slice: slice,
    control_slice: slice,
    loss_slice: slice,
) -> torch.Tensor:
    vocab_size = model.get_input_embeddings().weight.shape[0]
    control_ids = input_ids[control_slice]

    e = F.one_hot(control_ids, vocab_size).float().requires_grad_(True)
    embed_weight = model.get_input_embeddings().weight

    with torch.enable_grad():
        all_embeds = model.get_input_embeddings()(input_ids.unsqueeze(0))
        control_embeds = (e @ embed_weight).unsqueeze(0)
        all_embeds = all_embeds.clone()
        all_embeds[0, control_slice] = control_embeds[0]

        out = model(inputs_embeds=all_embeds)
        logits = out.logits[0]

        target_logits = logits[loss_slice]
        target_ids = input_ids[target_slice]
        loss = F.cross_entropy(target_logits, target_ids)
        loss.backward()

    grad = e.grad
    if torch.any(torch.isnan(grad)):
        nan_pos = torch.where(torch.any(torch.isnan(grad), dim=-1))[0][0].item()
        raise GCGNaNGradientError(step=0, position=int(nan_pos))
    return grad.detach()


def top_k_candidates(
    grad: torch.Tensor,
    input_ids: torch.Tensor,
    control_slice: slice,
    topk: int,
    not_allowed_tokens: set[int] | None = None,
) -> torch.Tensor:
    neg_grad = -grad.clone()
    if not_allowed_tokens:
        for token_id in not_allowed_tokens:
            if token_id < neg_grad.shape[-1]:
                neg_grad[:, token_id] = float("-inf")
    _, top_indices = torch.topk(neg_grad, topk, dim=-1)
    return top_indices


def sample_candidates(
    top_k_ids: torch.Tensor,
    batch_size: int,
    rng: torch.Generator,
) -> torch.Tensor:
    control_len, topk = top_k_ids.shape
    result = torch.zeros(batch_size, control_len, dtype=torch.long)
    for i in range(control_len):
        indices = torch.multinomial(
            torch.ones(topk, device=top_k_ids.device),
            num_samples=batch_size,
            replacement=True,
            generator=rng,
        )
        result[:, i] = top_k_ids[i][indices]
    return result


def build_candidate_input_ids(
    input_ids: torch.Tensor,
    control_slice: slice,
    candidates: torch.Tensor,
) -> torch.Tensor:
    batch_size = candidates.shape[0]
    all_ids = input_ids.unsqueeze(0).expand(batch_size, -1).clone()
    all_ids[:, control_slice] = candidates
    return all_ids


def evaluate_candidates(
    model,
    candidate_input_ids: torch.Tensor,
    target_slice: slice,
    loss_slice: slice,
    eval_chunk_size: int = 64,
) -> torch.Tensor:
    all_losses = []
    for chunk in candidate_input_ids.split(eval_chunk_size):
        with torch.no_grad():
            out = model(input_ids=chunk, use_cache=False)
            logits = out.logits
        target_ids = chunk[:, target_slice]
        loss_logits = logits[:, loss_slice]
        losses = F.cross_entropy(
            loss_logits.reshape(-1, loss_logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape(chunk.shape[0], -1).mean(dim=-1)
        all_losses.append(losses)
    return torch.cat(all_losses)


def greedy_select(
    losses: torch.Tensor,
    candidates: torch.Tensor,
    current_control_ids: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    best_idx = losses.argmin().item()
    return candidates[best_idx], losses[best_idx].item()
