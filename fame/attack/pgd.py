"""The Projected Gradient Descent attack."""
from typing import Callable

import numpy as np
import torch
Tensor = torch.Tensor
from .fgsm import fast_gradient_method
from .utils import clip_eta, optimize_linear


def projected_gradient_descent(
    model_fn: torch.nn.Module,
    x: Tensor,
    eps: float,
    eps_iter: float,
    nb_iter: int,
    norm: int,
    loss_fn: Callable,
    clip_min: Tensor = None,
    clip_max: Tensor = None,
    y: Tensor = None,
    targeted: bool = False,
    rand_init: bool = True,
    rand_minmax: float = None,
    sanity_checks: bool = False,
) -> tuple[Tensor, list[Tensor]]:
    """Creates adversarial examples using Projected Gradient Descent (PGD).

    This function implements the PGD attack, a strong, iterative method for
    generating adversarial examples, as originally proposed by Madry et al.
    PGD is essentially a multi-step version of FGSM.

    The attack works by:
    1. Optionally starting from a random point within the allowed perturbation space.
    2. Iteratively taking small steps (using FGSM as the inner step) in the
       direction that maximizes the loss.
    3. After each step, projecting the total perturbation back onto the $L_p$
       ball of radius `eps` to ensure the constraint is not violated.
    4. Clipping the resulting example to a valid data range (e.g., [0, 1]).

    Reference:
    Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017).
    Towards deep learning models resistant to adversarial attacks.
    https://arxiv.org/abs/1706.06083
    Code: cleverhans

    Args:
        model_fn: A callable model function (e.g., a PyTorch Module) that takes
            a tensor input and returns the model's logits.
        x: The input tensor to be perturbed.
        eps: The total perturbation budget (radius of the $L_p$ ball).
        eps_iter: The step size for each attack iteration.
        nb_iter: The number of attack iterations to perform.
        norm: The norm order to use for the perturbation ($L_\infty$ or $L_2$).
            Accepts `np.inf` or `2`.
        loss_fn: The loss function used to compute the gradient.
        clip_min: An optional tensor for the minimum value to which the
            adversarial example will be clipped at each iteration.
        clip_max: An optional tensor for the maximum value to which the
            adversarial example will be clipped at each iteration.
        y: The ground-truth or target labels. If `None`, the model's own
            predictions on the clean input `x` are used.
        targeted: If `True`, performs a targeted attack. If `False` (default),
            the attack is untargeted.
        rand_init: If `True`, starts the attack from a random point within the
            epsilon ball, which can improve attack success.
        rand_minmax: The range (`-rand_minmax` to `+rand_minmax`) for the
            random initialization. Defaults to `eps`.
        sanity_checks: If `True`, performs assertions on inputs.

    Returns:
        A tuple containing:
        - `adv_x` (Tensor): The final adversarial example found after `nb_iter` steps.
        - `x_hist` (List[Tensor]): A list of adversarial examples from each
          iteration of the attack.
    """

    if norm == 1:
        raise NotImplementedError(
            "It's not clear that FGM is a good inner loop"
            " step for PGD when norm=1, because norm=1 FGM "
            " changes only one pixel at a time. We need "
            " to rigorously test a strong norm=1 PGD "
            "before enabling this feature."
        )
    if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")
    if eps < 0:
        raise ValueError("eps must be greater than or equal to 0, got {} instead".format(eps))
    if eps == 0:
        return x
    if eps_iter < 0:
        raise ValueError(
            "eps_iter must be greater than or equal to 0, got {} instead".format(eps_iter)
        )
    if eps_iter == 0:
        return x

    assert eps_iter <= eps, (eps_iter, eps)
    """
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )
    """

    asserts = []

    x_hist = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        #assert_ge = torch.all(torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype)))
        assert_ge = torch.all(torch.ge(x, clip_min.detach().clone().to(device=x.device, dtype=x.dtype)))
        asserts.append(assert_ge)

    if clip_max is not None:
        #assert_le = torch.all(torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype)))
        # Best practice if clip_max is a Tensor
        assert_le = torch.all(torch.le(x, clip_max.detach().clone().to(device=x.device, dtype=x.dtype)))
        asserts.append(assert_le)

    # Initialize loop variables
    if rand_init:
        if rand_minmax is None:
            rand_minmax = eps
        eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)
    else:
        eta = torch.zeros_like(x)

    # Clip eta
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)

    i = 0
    while i < nb_iter:
        adv_x = fast_gradient_method(
            model_fn,
            adv_x,
            eps_iter,
            norm,
            loss_fn=loss_fn,
            clip_min=clip_min,
            clip_max=clip_max,
            y=y,
            targeted=targeted,
        )

        # Clipping perturbation eta to norm norm ball
        eta = adv_x - x
        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta

        # Redo the clipping.
        # FGM already did it, but subtracting and re-adding eta can add some
        # small numerical error.
        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)
        i += 1
        x_hist.append(adv_x)

    asserts.append(eps_iter <= eps)
    if norm == np.inf and clip_min is not None:
        asserts.append(eps + clip_min <= clip_max)

    if sanity_checks:
        assert np.all(asserts)

    return adv_x, x_hist
