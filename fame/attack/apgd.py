"""The Auto Projected Gradient Descent (APGD) attack."""
from typing import Callable, Tuple, List, Optional
import numpy as np
import torch

Tensor = torch.Tensor
from .utils import clip_eta


def autopgd_attack(
    model_fn: torch.nn.Module,
    x: Tensor,
    eps: float,
    nb_iter: int = 100,
    norm: int = np.inf,
    loss_fn: Optional[Callable] = None,
    clip_min: Optional[Tensor] = None,
    clip_max: Optional[Tensor] = None,
    y: Optional[Tensor] = None,
    targeted: bool = False,
    rho: float = 0.75,
    n_restarts: int = 1,
) -> Tuple[Tensor, List[Tensor]]:
    """Creates adversarial examples using AutoPGD (APGD).

    APGD automatically tunes the step size during the optimization process,
    reducing step size when loss fails to increase consistently.

    Reference:
    Croce, F., & Hein, M. (2020).
    Reliable evaluation of adversarial robustness with an ensemble of diverse attacks.
    ICML. https://arxiv.org/abs/2003.01690
    """
    if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")
    if eps <= 0:
        return x, []

    device = x.device
    batch_size = x.shape[0]

    if y is None:
        with torch.no_grad():
            y = model_fn(x).argmax(dim=1)

    # Loss function setup (CrossEntropy by default)
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    # Define check points for step size adaptation
    w = [0]
    p = 0.22
    while p < 1.0:
        w.append(int(p * nb_iter))
        p += max(p - w[-2] / nb_iter - 0.03, 0.06)
    w.append(nb_iter)

    best_adv_x = x.clone().detach()
    best_loss = torch.full((batch_size,), -float("inf"), device=device)
    x_hist = []

    for _ in range(n_restarts):
        # 1. Random initialization inside the eps-ball
        if norm == np.inf:
            eta = torch.zeros_like(x).uniform_(-eps, eps)
        else:
            eta = torch.randn_like(x)
            eta_flat = eta.view(batch_size, -1)
            eta_norm = eta_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            r = torch.zeros((batch_size, 1), device=device).uniform_(0, eps)
            eta = (eta_flat / eta_norm * r).view_as(x)

        adv_x = x + clip_eta(eta, norm, eps)
        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)

        adv_x_best = adv_x.clone().detach()
        alpha = 2.0  # Initial step-size factor
        loss_best = torch.full((batch_size,), -float("inf"), device=device)
        loss_checkpoint = torch.zeros(batch_size, device=device)
        
        # Momentum initialization
        u = torch.zeros_like(x)
        eta_w = 0.75

        for i in range(nb_iter):
            adv_x.requires_grad_()
            logits = model_fn(adv_x)
            
            # Compute loss and gradients
            loss = loss_fn(logits, y)
            if targeted:
                loss = -loss
                
            grad = torch.autograd.grad(loss.sum(), adv_x)[0]

            # Update best-found adversarial points per sample
            mask = loss > loss_best
            loss_best[mask] = loss[mask].detach()
            adv_x_best[mask] = adv_x[mask].detach()

            # Tracking global best loss across restarts
            global_mask = loss > best_loss
            best_loss[global_mask] = loss[global_mask].detach()
            best_adv_x[global_mask] = adv_x[global_mask].detach()

            x_hist.append(best_adv_x.clone().detach())

            with torch.no_grad():
                # Normalized gradient step calculation
                if norm == np.inf:
                    step_dir = grad.sign()
                elif norm == 2:
                    grad_flat = grad.view(batch_size, -1)
                    grad_norm = grad_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                    step_dir = (grad_flat / grad_norm).view_as(grad)

                # Step update with momentum term
                eta_next = adv_x + alpha * eps * step_dir - x
                eta_next = clip_eta(eta_next, norm, eps)
                z = x + eta_next

                # Apply momentum equation: x^(i+1) = P(x^(i) + eta_w * (z - x^(i)) + (1 - eta_w) * (x^(i) - x^(i-1)))
                adv_x = adv_x + eta_w * (z - adv_x) + (1.0 - eta_w) * u
                u = adv_x - z

                # Projection & domain clipping
                eta = adv_x - x
                eta = clip_eta(eta, norm, eps)
                adv_x = x + eta
                if clip_min is not None or clip_max is not None:
                    adv_x = torch.clamp(adv_x, clip_min, clip_max)

                # Checkpoint adaptation condition
                if i + 1 in w:
                    # Evaluate condition 1: fraction of iterations loss improved
                    c1 = (loss_best > loss_checkpoint).float().mean()
                    
                    # Condition 2: step size reduction trigger
                    if c1 < rho:
                        alpha /= 2.0
                        adv_x = adv_x_best.clone()  # Restart from best local point

                    loss_checkpoint = loss_best.clone()

    return best_adv_x, x_hist