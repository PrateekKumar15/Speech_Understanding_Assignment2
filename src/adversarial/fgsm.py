from __future__ import annotations

import torch


def snr_db(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    signal = clean.pow(2).mean().item()
    noise = (noisy - clean).pow(2).mean().item()
    noise = max(noise, 1e-12)
    return 10.0 * torch.log10(torch.tensor(signal / noise)).item()


def fgsm_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    clamp: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = torch.nn.functional.cross_entropy(logits, y)
    model.zero_grad(set_to_none=True)
    loss.backward()

    grad_sign = x_adv.grad.sign()
    perturbed = x_adv + epsilon * grad_sign
    perturbed = torch.clamp(perturbed, clamp[0], clamp[1])
    return perturbed.detach()


def find_min_epsilon_for_flip(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    eps_values: list[float],
    min_snr_db: float = 40.0,
) -> tuple[float | None, float | None]:
    model.eval()

    with torch.no_grad():
        pred = model(x).argmax(dim=-1)
    if int(pred.item()) != int(y_true.item()):
        return 0.0, None

    for eps in eps_values:
        x_adv = fgsm_attack(model, x, y_true, epsilon=eps)
        snr = snr_db(x, x_adv)
        with torch.no_grad():
            pred_adv = model(x_adv).argmax(dim=-1)

        if int(pred_adv.item()) != int(y_true.item()) and snr >= min_snr_db:
            return eps, snr

    return None, None
