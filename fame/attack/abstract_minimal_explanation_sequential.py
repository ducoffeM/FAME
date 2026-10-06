import torch
import numpy as np
from fame.batch_free.order import get_greedy_order
Tensor = torch.Tensor

from .attack import attack, find_singleton_feature_2_add
from .fgsm import fast_gradient_method
from .pgd import projected_gradient_descent


def find_closest_xai(
    model: torch.nn.Module,
    gt_label: int,
    input_sample: Tensor,
    lower_bound: Tensor,
    upper_bound: Tensor,
    eps: float = 0.0,
    xai_indices: list[int] = [],
    free_indices: list[int] = [],
    method: str = "fgsm",
    device: str = "mps",
    channel: int = 1,
    data_format: int = "channels_first",
    n_class: int = 10,
    traversal_order: str = "greedy",
    verbose:int=0,
    singleton_only:bool=False, 
) -> tuple[list[int], list[int]]:
    """Identifies a minimal set of robust features required to prevent adversarial attacks.

    This function searches for a partition of input features into two sets: a set of
    "vulnerable" features and a minimal set of "robust" features. Fixing the
    robust set is sufficient to ensure the model's prediction remains correct
    against attacks on the vulnerable set. This can be interpreted as finding a
    concise explanation for the model's robustness (or lack thereof).

    The algorithm operates as follows:
    1.  It first checks if an adversarial attack is possible when all features
        are perturbed. If not, the search is skipped.
    2.  It identifies any features that can cause a misclassification on their own.
    3.  It then iteratively builds up the robust set by adding features one by
        one (in a greedy order of influence). After adding each feature, it
        re-evaluates the remaining features to see if they have now become
        vulnerable in this new context.

    Args:
        model: The Keras model to be analyzed.
        gt_label: The ground-truth class label to defend.
        input_sample: A single input point serving as the center of the
            perturbation.
        eps: The radius of the $L_\infty$ perturbation.
        xai_indices: A list of pre-defined XAI feature indices, excluded from
            the search.
        free_indices: A list of feature indices that are always considered
            perturbed.
        method: The adversarial attack algorithm to use (e.g., "fgsm").
        device: The computational device to run on (e.g., "mps", "cuda").
        channel: The number of channels in the input data.
        data_format: The data format, either "channels_first" or "channels_last".
        n_class: The number of output classes of the model.
        traversal_order: The strategy for iterating through features. Currently,
            only "greedy" is supported.

    Returns:
        A tuple containing two lists of feature indices:
        - **vulnerable_features** (list[int]): The set of features that can
          contribute to a successful adversarial attack.
        - **robust_subset** (list[int]): The minimal set of features that, when
          considered robust (i.e., perturbed alongside `free_indices`),
          are sufficient to guarantee the model's correctness against attacks on
          the `vulnerable_features`.
    """

    n_in_wo_channel: int = int(input_sample.shape[-1] / channel)


    # start by attacking everything
    adv_pred_everything: np.array = attack(
        model=model,
        input_sample_batch=input_sample[None],
        lower_bound_batch=lower_bound[None],
        upper_bound_batch=upper_bound[None],
        gt_label=gt_label,
        eps=eps,
        method=method,
        device=device
    )  # (1,)
    if adv_pred_everything[0] == gt_label:
        print("no attacks could be find, skip the search !!!!")
        return [], [i for i in range(n_in_wo_channel) if not i in xai_indices + free_indices]

    # compute traversal order
    if traversal_order == "lirpa":
        
        remaining_features_with_traversal = get_greedy_order(
            model=model,
            input_sample=input_sample,
            gt_label=gt_label,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            free_indices=free_indices,
            channel=channel,
            data_format=data_format,
            n_class=n_class,
        )
    elif traversal_order=="arrange":
        remaining_features_with_traversal = [i for i in range(n_in_wo_channel) if not i in xai_indices + free_indices]
    else:
        raise NotImplementedError("implement other orders if needed")

    remaining_indices: list[int] = [
        i for i in range(n_in_wo_channel) if not i in free_indices + xai_indices
    ]
    N: int = len(remaining_indices)
    # start with the current set of free indices
    # indices that could be attacked found so far

    potential_xai = find_singleton_feature_2_add(
        model=model,
        gt_label=gt_label,
        input_sample=input_sample,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        eps=eps,
        free_indices=free_indices,
        remaining_indices=remaining_indices,
        method=method,
        device=device,
        channel=channel,
        data_format=data_format,
    )


     
    assert all(idx in remaining_indices for idx in potential_xai), "potential_xai should be a subset of remaining_indices"

    # collapse lower and upper bound of potential_xai

    # remove attackable dimensions from remaining_indices
    remaining_indices = [i for i in remaining_indices if not i in potential_xai]
    extra_free = []
    for index in remaining_features_with_traversal:
        if index in potential_xai:
            continue

        extra_free.append(index)
        remaining_indices = [i for i in remaining_indices if not i in potential_xai + [index]]

        if len(remaining_indices):
            potential_xai_j = find_singleton_feature_2_add(
                model=model,
                gt_label=gt_label,
                input_sample=input_sample,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                eps=eps,
                free_indices=free_indices + extra_free,
                remaining_indices=remaining_indices,
                method=method,
                device=device,
                channel=channel,
                data_format=data_format,
            )
            assert all(idx in remaining_indices for idx in potential_xai_j), "potential_xai_j should be a subset of remaining_indices"
            potential_xai += potential_xai_j
            remaining_indices = [i for i in remaining_indices if not i in potential_xai]
        else:
            # no more features to free
            break

    assert len(potential_xai + extra_free) == N, "ValueError"

    return potential_xai, extra_free
