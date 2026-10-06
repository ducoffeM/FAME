"""The Fast Gradient Method attack."""
from typing import Callable

import numpy as np
import torch
Tensor = torch.Tensor
from .utils import optimize_linear


def fast_gradient_method(
    model_fn: torch.nn.Module,
    x: Tensor,
    eps: float,
    norm: int,
    loss_fn: Callable,
    clip_min: Tensor = None,
    clip_max: Tensor = None,
    y: Tensor = None,
    targeted: bool = False,
    sanity_checks: bool = False,
) -> Tensor:
    """Creates adversarial examples using the Fast Gradient Sign Method (FGSM).

    This function implements the FGSM attack, a single-step gradient-based method
    for generating adversarial examples, as originally proposed by Goodfellow et al.
    The attack works by taking a single step in the direction of the gradient of
    the loss function with respect to the input data.

    The perturbation is calculated to maximize the loss (for untargeted attacks)
    or minimize it (for targeted attacks) within an $L_p$ ball of radius epsilon.

    Reference:
    Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014).
    Explaining and harnessing adversarial examples.
    https://arxiv.org/abs/1412.6572
    Code References: cleverhans


    Args:
        model_fn: A callable model function (e.g., a PyTorch Module) that takes
            a tensor input and returns the model's logits.
        x: The input tensor to be perturbed.
        eps: The magnitude of the perturbation (epsilon), controlling the
            attack strength within the specified norm.
        norm: The norm order to use for the perturbation ($L_\infty$, $L_1$, or $L_2$).
            Accepts `np.inf`, `1`, or `2`.
        loss_fn: The loss function used to compute the gradient (e.g.,
            `torch.nn.CrossEntropyLoss`).
        clip_min: An optional tensor for the minimum value to which the
            adversarial example will be clipped.
        clip_max: An optional tensor for the maximum value to which the
            adversarial example will be clipped.
        y: The ground-truth or target labels. If `None`, the model's own
            predictions on the clean input `x` are used.
        targeted: If `True`, the attack is targeted and aims to misclassify the
            input as the class specified in `y`. If `False` (default), the
            attack is untargeted and aims to maximize the loss for the
            correct class.
        sanity_checks: If `True`, performs assertions to check that the input `x`
            is within the `[clip_min, clip_max]` range.

    Returns:
        An adversarial tensor of the same shape as `x` that has been perturbed
        to fool the model.
    """

    if norm not in [np.inf, 1, 2]:
        raise ValueError("Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm))
    if eps < 0:
        raise ValueError("eps must be greater than or equal to 0, got {} instead".format(eps))
    if eps == 0:
        return x

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype)))
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype)))
        asserts.append(assert_le)

    # x needs to be a leaf variable, of floating point type and have requires_grad being True for
    # its grad to be computed and stored properly in a backward call
    x = x.clone().detach().to(torch.float).requires_grad_(True)
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)

    # Compute loss
    loss = loss_fn(model_fn(x), y)
    # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    loss.backward()
    optimal_perturbation = optimize_linear(x.grad, eps, norm)

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        if clip_min is None or clip_max is None:
            raise ValueError(
                "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
            )
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x
