import numpy as np
import torch
Tensor = torch.Tensor
from .fgsm import fast_gradient_method
from .pgd import projected_gradient_descent
from .apgd import autopgd_attack
from .utils import get_attacks_bounds


def attack(
    model: torch.nn.Module,
    input_sample_batch: np.ndarray,
    lower_bound_batch: np.ndarray,
    upper_bound_batch: np.ndarray,
    gt_label: int,
    eps: float,
    method: str,
    device: str = "cuda",
) -> np.ndarray:
    """Generates adversarial examples for a batch of inputs using a specified method.

    This function is a wrapper that performs adversarial attacks (either FGSM or PGD)
    on a batch of inputs. It takes a Keras model and numpy arrays, converts them
    to PyTorch tensors to leverage attack libraries, generates the adversarial
    examples, and returns the model's predictions on them as a numpy array.

    The generated adversarial examples are constrained to lie within the provided
    per-sample lower and upper bounds.

    Args:
        model: The Keras model to attack. It will be temporarily converted for use
            with PyTorch.
        input_sample_batch: A batch of nominal input samples.
        lower_bound_batch: A batch of lower bound arrays, defining the clipping
            region for each sample.
        upper_bound_batch: A batch of upper bound arrays, defining the clipping
            region for each sample.
        gt_label: The ground-truth label, used as the target for the attack
            (i.e., the attack tries to move the prediction away from this label).
        eps: The perturbation budget (epsilon) for the $L_\infty$ norm.
        method: The attack algorithm to use. Supported options are "fgsm" and "pgd".
        device: The computation device (e.g., "mps", "cuda", "cpu").

    Returns:
        A numpy array containing the model's predicted class labels for the
        generated adversarial examples.
    """

    model.to(device)
    input_t: Tensor = torch.tensor(input_sample_batch, dtype=torch.float32, device=device)
    lower_t: Tensor = torch.tensor(lower_bound_batch, dtype=torch.float32, device=device)
    upper_t: Tensor = torch.tensor(upper_bound_batch, dtype=torch.float32, device=device)
    batch_size = len(input_sample_batch)
    gt_label_t: Tensor = torch.tensor(
        np.array([gt_label] * batch_size, dtype="int64"), dtype=torch.int64, device=device
    )

    if method == "fgsm":
        x_adv_class = fast_gradient_method(
            model_fn=model,
            x=input_t,
            eps=eps,
            loss_fn=torch.nn.CrossEntropyLoss(),
            norm=np.inf,
            clip_min=lower_t,
            clip_max=upper_t,
            y=gt_label_t,
        )
    elif method=="pgd":
        eps_iter = eps / 10.0  # arbitraty (do a config file)
        nb_iter = 10  # arbitraty (do a config file)
        x_adv_class, _ = projected_gradient_descent(
            model_fn=model,
            x=input_t,
            eps=eps,
            eps_iter=eps_iter,
            nb_iter=nb_iter,
            loss_fn=torch.nn.CrossEntropyLoss(),
            norm=np.inf,
            clip_min=lower_t,
            clip_max=upper_t,
            y=gt_label_t,
            rand_init=False,  # for reproducibility
        )
    elif method=="apgd":
        nb_iter = 50  # Number of APGD iterations

        x_adv_class, _ = autopgd_attack(
            model_fn=model,
            x=input_t,
            eps=eps,
            nb_iter=nb_iter,
            norm=np.inf,
            loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),  # APGD uses per-sample loss
            clip_min=lower_t,
            clip_max=upper_t,
            y=gt_label_t,
            n_restarts=1,  # Single run (replaces rand_init)    
        )

    else:
        raise NotImplementedError('unknow attack method {}'.format(method))
    adv_pred = model(x_adv_class).argmax(-1)

    return adv_pred  # (batch_size,)


def find_singleton_feature_2_add(
    model: torch.nn.Module,
    gt_label: int,
    input_sample: Tensor,
    lower_bound: Tensor,
    upper_bound: Tensor,
    eps: float = 0.0,
    free_indices: list[int] = [],
    remaining_indices: list[int] = [],  # attack only those features
    method: str = "fgsm",
    device: str = "cuda",
    channel: int = 1,
    data_format="channels_first",
) -> list[int]:
    """Identifies individual features that can cause an adversarial attack.

    This function tests the vulnerability of each feature in `remaining_indices`
    one by one. For each feature, it constructs a perturbation domain where a set of
    `free_indices` are always allowed to be perturbed, and additionally, the
    single feature under test is also allowed to be perturbed.

    It then runs an adversarial attack within each of these constructed domains.
    The function returns a list of all features from `remaining_indices` that
    were sufficient to cause a misclassification in their respective test.

    Args:
        model: The Keras model to be analyzed.
        gt_label: The correct label to check for misclassification against.
        input_sample: The single, nominal input point.
        eps: The radius of the $L_\infty$ perturbation.
        free_indices: A list of feature indices that are always considered
            perturbed in the background of every test.
        remaining_indices: The pool of features to test individually for their
            ability to cause an attack.
        method: The adversarial attack algorithm to use.
        device: The computation device for the attack.
        channel: The number of channels in the input data.
        data_format: The data format, either "channels_first" or "channels_last".

    Returns:
        A list of indices from `remaining_indices` corresponding to features
        that were found to be vulnerable.
    """

    model.to(device)
    model.eval()

    input_sample_batch, lower_bound_batch, upper_bound_batch = get_attacks_bounds(
        input_sample=input_sample,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        eps=eps,
        free_indices=free_indices,
        remaining_indices=remaining_indices,
        channel=channel,
        data_format=data_format,
    )
    batch_size: int = len(remaining_indices)

    adv_pred: np.ndarray = attack(
        model=model,
        input_sample_batch=input_sample_batch,
        lower_bound_batch=lower_bound_batch,
        upper_bound_batch=upper_bound_batch,
        gt_label=gt_label,
        eps=eps,
        method=method,
        device=device,
    )  # (batch_size,)

    xai_set: list[int] = [remaining_indices[j] for j in range(batch_size) if adv_pred[j] != gt_label]
    return xai_set
