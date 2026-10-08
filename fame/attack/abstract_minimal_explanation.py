import numpy as np
import torch
from fame.batch_free.order import get_greedy_order
Tensor = torch.Tensor
from .abstract_minimal_explanation_sequential import (
    find_closest_xai as find_closest_xai_singleton,
)
from .attack import attack, find_singleton_feature_2_add
from .utils import get_attacks_bounds


def find_closest_xai_with_dichotomy(
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
    v_optimal:bool=False,
    verbose:int=0
) -> tuple[list[int], list[int]]:
    """Finds a robust feature set using a recursive divide-and-conquer algorithm.

    This function implements a recursive search to find a small
    set of features that are sufficient to ensure the model's robustness. It
    partitions the input features into a "vulnerable" set and a "robust" set.

    The dichotomy (divide-and-conquer) strategy works as follows:
    1.  It establishes base cases for when the search space is empty or trivial.
    2.  It identifies features that are vulnerable on their own ("singletons").
    3.  For the remaining features, it recursively splits them into two halves
        based on a greedy ordering of influence (traversal order).
    4.  It analyzes the interaction between the two halves by checking if
        perturbing one half makes features in the other half vulnerable.
    5.  It greedily explores the path that appears to yield the smallest robust
        set, making recursive calls on smaller sub-problems.

    This method serves as a fast heuristic to approximate the minimal robust set,
    trading guaranteed optimality for a significant speed improvement.

    Args:
        model: The Keras model to be analyzed.
        gt_label: The ground-truth class label to defend.
        input_sample: A single input point serving as the center of the perturbation.
        eps: The radius of the $L_\infty$ perturbation.
        xai_indices: A list of pre-defined XAI features, excluded from the search.
        free_indices: A list of features that are always considered perturbed.
        method: The adversarial attack algorithm to use (e.g., "fgsm").
        device: The computational device to run on (e.g., "mps", "cuda").
        channel: The number of channels in the input data.
        data_format: The data format, either "channels_first" or "channels_last".
        n_class: The number of output classes of the model.
        traversal_order: The strategy for ordering features before splitting.
        v_optimal: A flag indicating whether to seek an optimal vulnerable set (see FAVEX).

    Returns:
        A tuple containing two lists of feature indices:
        - **vulnerable_features**: An approximation of the features that
          can contribute to adversarial examples.
        - **robust_subset**: An approximation of a minimal set of features
          that ensures robustness.
    """

    # compute attacks on every remaining feature
    n_in_wo_channel: int = int(input_sample.shape[-1] / channel)
    remaining_indices = [i for i in range(n_in_wo_channel) if not i in xai_indices + free_indices]

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
        return [], remaining_indices



    if len(remaining_indices) == 0:
        # best solution is
        assert len(xai_indices + free_indices) == n_in_wo_channel, "missing input features"
        return xai_indices, free_indices


    # start by attacking everything except xai_indices
    input_sample_everything, lower_bound_everything, upper_bound_everything = get_attacks_bounds(
        input_sample=input_sample,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        eps=eps,
        free_indices=free_indices,
        remaining_indices=remaining_indices,
        channel=channel,
        data_format=data_format,
    )


    xai_set_init = find_singleton_feature_2_add(
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
    assert set(xai_set_init).isdisjoint(set(free_indices)), "xai_set_init should not overlap with free_indices"

    if not v_optimal:
        return xai_set_init, remaining_indices
    else:
        raise NotImplementedError("v_optimal=True is not implemented yet for dichotomy search")



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
    v_optimal:bool=False,
    verbose:int=0,
) -> tuple[list[int], list[int]]:
    """Finds a minimal robust feature set using a two-stage search process.

    This function orchestrates a hybrid search strategy to efficiently and
    accurately partition input features into a "vulnerable" set and a "robust"
    set, providing a concise explanation for the model's behavior.

    The process is as follows:
    1.  **Dichotomy Search**: It first calls `find_closest_xai_with_dichotomy`
        to perform a fast, divide-and-conquer search. This quickly prunes the
        search space and identifies a broad set of potentially vulnerable features.
    2.  **Sequential Refinement**: It then uses a more thorough, sequential
        search method (`find_closest_xai_singleton`) on the candidate set from
        the first step. This refines the initial approximation to find a tighter
        and more accurate result.

    This two-stage approach balances the speed of heuristic search
    with the precision of a more exhaustive one.

    Args:
        model: The Keras model to be analyzed.
        gt_label: The ground-truth class label to defend.
        input_sample: A single input point serving as the center of the perturbation.
        eps: The radius of the $L_\infty$ perturbation.
        xai_indices: A list of pre-defined XAI features, excluded from the search.
        free_indices: A list of features that are always considered perturbed.
        method: The adversarial attack algorithm to use (e.g., "fgsm").
        device: The computational device to run on (e.g., "mps", "cuda").
        channel: The number of channels in the input data.
        data_format: The data format, either "channels_first" or "channels_last".
        n_class: The number of output classes of the model.
        traversal_order: The strategy for iterating through features.
        v_optimal: A flag indicating whether to seek an optimal vulnerable set (see FAVEX).
    Returns:
        A tuple containing two lists of feature indices from the refined search:
        - **vulnerable_features**: The final set of features that can
          contribute to a successful adversarial attack.
        - **robust_subset**: The final minimal set of features that ensures
          robustness.
    """

    n_in_wo_channel: int = int(input_sample.shape[-1] / channel)
    potential_xai_d, _ = find_closest_xai_with_dichotomy(
        model=model,
        gt_label=gt_label,
        input_sample=input_sample,
        lower_bound= lower_bound,
        upper_bound= upper_bound,
        eps=eps,
        xai_indices=xai_indices,
        free_indices=free_indices,
        method=method,
        device=device,
        channel=channel,
        data_format=data_format,
        n_class=n_class,
        traversal_order=traversal_order,
        v_optimal=v_optimal
    )

    assert set(potential_xai_d).isdisjoint(set(free_indices)), "potential_xai_d should not overlap with free_indices"
    # we have already a minimal explanation
    if not v_optimal:
        return potential_xai_d, []
    else:
        raise NotImplementedError("v_optimal=True is not implemented yet for dichotomy search")
