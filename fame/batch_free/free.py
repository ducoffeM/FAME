from typing import Tuple
import torch
import numpy as np
from auto_LiRPA.perturbations import Perturbation, PerturbationLpNorm
import torch.nn as nn
from typing import TypeVar

# Type alias: 'Model' can now be used in any type hint
Model = nn.Module
Tensor = torch.Tensor

# Generic TypeVar constrained to nn.Module (useful for generic functions)
ModelType = TypeVar("ModelType", bound=nn.Module)


from fame.abstract_domain.abstract import (
    get_abstract_model, get_abstract_output_domain
)

from fame.abstract_domain.cardinality_domain import XAIDomain
from fame.batch_free.utils import encode_matrix

from .greedy import get_greedy
from .singleton import free_with_binary_search
from .utils import get_b, get_free_mask, get_W, get_xai_mask

def get_features_batch(
    model: Model,
    input_shape: Tuple[int],
    gt_label: int,
    input_sample: Tensor, # (?, *input_shape)
    lower_bound_input: Tensor,
    upper_bound_input: Tensor,
    free_indices: list[int],
    cardinality: np.ndarray,
    data_format: str = "channels_first",
    n_class: int = 10,
    batch_size: int = 1,
    method:str='CROWN',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Performs a batched abstract interpretation pass over a hybrid L-inf/L0 domain.

    This function is a low-level wrapper around `decomon` that analyzes a next calllex
    perturbation space in a single, batched forward pass. It constructs a
    specialized `XAIDomain` which models a hybrid perturbation where:
    1. A set of `free_indices` are always perturbed within an L-infinity ball.
    2. Up to `k` features (defined by `cardinality`) from the `xai_indices`
       pool can also be perturbed (L0-norm constraint).

    It then performs backward bound propagation (CROWN) to extract both the final
    concrete upper bounds and the parameters of the final affine relaxation.

    Args:
        model: The pytorch model to analyze.
        gt_label: The ground-truth class label.
        input_sample: The nominal input point.
        lower_bound_input: The lower bounds of the L-infinity perturbation.
        upper_bound_input: The upper bounds of the L-infinity perturbation.
        xai_indices: A list of feature indices subject to the L0 constraint.
        free_indices: A list of feature indices always perturbed under L-infinity.
        cardinality: A numpy array where each element specifies the L0 budget (`k`)
            for the corresponding item in the batch.
        channel: The number of channels in the input data.
        data_format: The data format, "channels_first" or "channels_last".
        n_class: The number of output classes of the model.

    Returns:
        A tuple of four numpy arrays:
        - `w_u`: The weights of the final affine upper bound with respect to the
          input features, shape `(batch, n_features, n_outputs)`.
        - `b_u`: The bias of the final affine upper bound, shape `(batch, n_outputs)`.
        - `upper`: The concrete (IBP) upper bounds on the logit differences.
        - `box`: The input domain tensor `[lower, upper, center]` used for analysis.
    """
    if data_format == "channels_first":
        channel, H, W = input_shape
    else:
        H, W, channel = input_shape

    n_in_with_channel: int = np.prod(input_shape)
    n_in_wo_channel: int = int(n_in_with_channel / channel)
    if (batch_size < 0) or (batch_size > len(cardinality)):
        batch_size = len(cardinality)
    all_w_u = []
    all_b_u = []
    all_upper = []
    all_box = []

    free_mask: np.ndarray = np.reshape(get_free_mask(n_in_wo_channel, free_indices), (H, W))  # (H, W)

    # modify the model to use asinput (batch, n_in_wo_channel, channel)
    device = next(model.parameters()).device
    lirpa_model = get_abstract_model(model=model,dummy_input=torch.zeros((1, *input_shape)).to(device))
    
    xai_perturbation_domain: PerturbationDomain=None
    for k in range(0, len(cardinality), batch_size):

        k_stop = min(k+batch_size, len(cardinality))
        current_batch_size = k_stop - k
        # lower_bound/upper_bound/input_sample.shape = (n_in,)
        lower_bound_batch: np.ndarray = np.repeat(
            np.copy(lower_bound_input[None]) + 0.0, repeats=current_batch_size, axis=0
        )  # (current_batch_size, channel, n_in_wo_channel)
        upper_bound_batch: np.ndarray = np.repeat(
            np.copy(upper_bound_input[None]) + 0.0, repeats=current_batch_size, axis=0
        )  # (current_batch_size, channel, n_in_wo_channel)
        input_sample_batch: np.ndarray = np.repeat(
            np.copy(input_sample[None] + 0.0), repeats=current_batch_size, axis=0
        )  # (current_batch_size, channel, n_in_wo_channel)

        box: np.ndarray = np.concatenate(
            [lower_bound_batch[:, None], upper_bound_batch[:, None], input_sample_batch[:, None]], 1
        )  # (current_batch_size, 3, channel, n_in_wo_channel)
        box = np.reshape(box, (current_batch_size, 3, -1))


        # convert to torch tensor
        device = next(model.parameters()).device
        lower_bound_batch: Tensor = torch.tensor(lower_bound_batch, dtype=torch.float32).to(device)
        upper_bound_batch: Tensor = torch.tensor(upper_bound_batch, dtype=torch.float32).to(device)
        input_sample_batch: Tensor = torch.tensor(input_sample_batch, dtype=torch.float32).to(device)

        # reshape to (batch_size, *input_shape)
        lower_bound_batch = lower_bound_batch.view(-1, *input_shape) # (current_batch_size, channel, H, W)
        upper_bound_batch = upper_bound_batch.view(-1, *input_shape) # (current_batch_size, channel, H, W)
        input_sample_batch = input_sample_batch.view(-1, *input_shape) # (current_batch_size, channel, H, W)


        # step 1: build your PerturbationDomain
        device = next(model.parameters()).device

        if xai_perturbation_domain is None:
            xai_perturbation_domain = XAIDomain(
                free_mask=torch.tensor(free_mask, dtype=torch.float32).to(device),
                eps=torch.tensor(cardinality[k:k_stop], dtype=torch.float32).to(device),
                data_format=data_format,
                x_L=lower_bound_batch,
                x_U=upper_bound_batch,
            )
        else:
            xai_perturbation_domain.eps_ = torch.tensor(cardinality[k:k_stop], dtype=torch.float32).to(device)
            xai_perturbation_domain.x_L = lower_bound_batch
            xai_perturbation_domain.x_U = upper_bound_batch


        # build your input domain
        # encode matrix C
        C_gt: np.ndarray = np.repeat(
            encode_matrix(n_class=n_class, groundtruth=gt_label)[None], repeats=current_batch_size, axis=0
        )  # (current_batch_size, n_class, n_class-1)
        C_gt = np.transpose(C_gt, (0, 2, 1))  # (current_batch_size, n_class-1, n_class)
        # convert to torch tensor
        device = next(model.parameters()).device
        C_gt: Tensor = torch.tensor(C_gt, dtype=torch.float32).to(device)

        w_u: np.ndarray
        b_u: np.ndarray
        upper: np.ndarray

        w_u, b_u, upper = get_abstract_output_domain(model=model, input_sample=input_sample_batch, \
                                                    C=C_gt,perturbation=xai_perturbation_domain, \
                                                    method=method, lirpa_model=lirpa_model)
        

        #w_u (current_batch_size, n_in_with_channel, n_class-1)
        #b_u = (current_batch_size, n_class-1)
        #upper = (current_batch_size, n_class-1)


        all_w_u.append(w_u)
        all_b_u.append(b_u)
        all_upper.append(upper)
        all_box.append(box)

    w_u = np.concatenate(all_w_u, axis=0) # (|cardinality|, n_in_with_channel, n_class-1)
    b_u = np.concatenate(all_b_u, axis=0) # (|cardinality|, n_class-1)
    upper = np.concatenate(all_upper, axis=0) # (|cardinality|, n_class-1)
    box = np.concatenate(all_box, axis=0) # (|cardinality|, 3, n_in_with_channel)

    return w_u, b_u, upper, box


def free_at_once_k_features(
    model: Model,
    input_shape: Tuple[int],
    gt_label: int,
    input_sample: np.ndarray,
    lower_bound_input: np.ndarray,
    upper_bound_input: np.ndarray,
    xai_indices: list[int] = [],
    free_indices: list[int] = [],
    cardinality: np.ndarray = None,
    data_format: str = "channels_first",
    n_class: int = 10,
    method: str = "greedy",
    lirpa_method="CROWN",
    batch_size_free = 1,
    verbose: int = 0,
) -> np.ndarray:
    """Finds the largest safe set of features, given cardinality constraints.

    This function attempts to find the largest set of features that can be added
    to the "robust set" (i.e., allowed to be perturbed) without violating the
    model's overall robustness. It solves this for a batch of different L0-norm
    cardinality constraints (`k`).

    The process involves:
    1.  Using abstract interpretation (`get_features_batch`) to obtain an affine
        approximation of the network's output logits.
    2.  Formulating a 0-1 Knapsack-like optimization problem from this approximation.
        The goal is to select the maximum number of features to "free" while
        ensuring the certified upper bound on logit differences remains non-positive.
    3.  Solving this problem using either an exact MILP solver or a fast greedy heuristic.

    Args:
        pytorch_model: The pytorch model to analyze.
        gt_label: The ground-truth class label.
        input_sample: The nominal input point.
        lower_bound_input: The lower bounds of the L-infinity perturbation.
        upper_bound_input: The upper bounds of the L-infinity perturbation.
        xai_indices: Candidate features for the knapsack problem.
        free_indices: Features that are always considered robust/perturbed.
        cardinality: A numpy array where each element specifies the L0 budget (`k`)
            for the corresponding item in the batch.
        channel: The number of channels in the input data.
        data_format: The data format, "channels_first" or "channels_last".
        n_class: The number of output classes.
        method: The optimization method to use: "milp" for an exact solution or
            "greedy" for a fast approximation.
        verbose: Verbosity level.

    Returns:
        A numpy array of shape `(batch_size, n_features)`, where each row is a
        binary mask indicating the set of features that can be safely made robust
        for the corresponding cardinality constraint.
    """
    lower_bound: np.ndarray = np.copy(lower_bound_input) + 0.0
    upper_bound: np.ndarray = np.copy(upper_bound_input) + 0.0

    if data_format == "channels_first":
        n_in_wo_channel: int = input_sample.shape[-1]
        channel: int = input_sample.shape[0]
    else:
        n_in_wo_channel: int = input_sample.shape[0] 
        channel: int = input_sample.shape[-1]


    input_: np.ndarray
    if data_format == "channels_first":
        lower_bound = np.reshape(lower_bound, (channel, n_in_wo_channel))  # (channel, n_in_wo_channel)
        upper_bound = np.reshape(upper_bound, (channel, n_in_wo_channel))
        input_: np.ndarray = np.reshape(np.copy(input_sample) + 0.0, (channel, n_in_wo_channel))
    else:
        lower_bound = np.reshape(lower_bound, (n_in_wo_channel, channel))  # (n_in_wo_channel, channel)
        upper_bound = np.reshape(upper_bound, (n_in_wo_channel, channel))
        input_: np.ndarray = np.reshape(np.copy(input_sample) + 0.0, (n_in_wo_channel, channel))

    # freeze xai features to the nominal value
    if len(xai_indices):
        if data_format == "channels_first":
            lower_bound[:, xai_indices] = input_[:, xai_indices]
            upper_bound[:, xai_indices] = input_[:, xai_indices]
        else:
            lower_bound[xai_indices, :] = input_[xai_indices, :]
            upper_bound[xai_indices, :] = input_[xai_indices, :]

    # reflatten
    batch_size: int = len(cardinality)
    w_u: np.ndarray
    b_u: np.ndarray
    box: np.ndarray

    w_u, b_u, upper, box = get_features_batch(
        model=model,
        input_shape=input_shape,
        gt_label=gt_label,
        input_sample=input_, # (n_in_wo_channel, channel) or (channel, n_in_wo_channel)
        lower_bound_input=lower_bound,
        upper_bound_input=upper_bound,
        free_indices=free_indices,
        cardinality=cardinality,
        data_format=data_format,
        n_class=n_class,
        method=lirpa_method,
        batch_size=batch_size_free
    )


    # w_u (batch_size, n_in_with_channel, n_class-1)
    # b_u (batch_size, n_class-1)
    # upper (batch_size, n_class-1)
    # box (batch_size, 3, n_in_with_channel)

    abstract_free_set: np.ndarray = np.zeros((batch_size, n_in_wo_channel))  # fill with 1 if in the solution

    if verbose:
        print("upper", upper)
        print("b", b_u)

    w_u_pos: np.ndarray = np.maximum(w_u, 0.0)  # (batch_size, n_in_with_channel, n_class-1)
    w_u_neg: np.ndarray = np.minimum(w_u, 0.0)  # (batch_size, n_in_with_channel, n_class-1)

    card_index: np.ndarray = np.array([i for i in np.arange(batch_size) if np.max(upper[i]) > 0])
    # trivial case, return the whole set given traversal order and cardinality constraints
    card_trivial: np.ndarray = np.array([i for i in np.arange(batch_size) if np.max(upper[i]) <= 0])

    xai_mask: np.ndarray = get_xai_mask(n_in_wo_channel, xai_indices)  # (1, n_in_wo_channel, 1)
    free_mask: np.ndarray = get_free_mask(n_in_wo_channel, free_indices)  # (1, n_in_wo_channel, 1)

    if len(card_trivial):
        # we could return all indices up to cardinalities
        # best is to return the highest one from the abstract domain (to facilitate freeing latter one)

        # we keep only indices from card_trivial
        w_u_trivial: np.ndarray = w_u[card_trivial]  # (|card_trivial|, n_in_with_channel, n_out)
        b_u_trivial: np.ndarray = b_u[card_trivial]  # (|card_trivial|, n_out)
        w_u_pos_trivial: np.ndarray = w_u_pos[
            card_trivial
        ]  # (|card_trivial|, n_in_with_channel, n_out)
        w_u_neg_trivial: np.ndarray = w_u_neg[
            card_trivial
        ]  # (|card_trivial|, n_in_with_channel, n_out)
        box_trivial: np.ndarray = box[card_trivial]  # (|card_trivial|, 3, n_in_with_channel)
        W_trivial: np.ndarray = get_W(
            w_u_pos_trivial, w_u_neg_trivial, box_trivial, channel=channel, data_format=data_format
        )  # (|card_trivial|, n_in_wo_channel, n_out)

        # set xai and free weights to zero (no impact)
        W_trivial = W_trivial * (1 - xai_mask) * (1 - free_mask)
        b_trivial: np.ndarray = get_b(
            W_trivial, w_u_trivial, b_u_trivial, box_trivial, free_mask
        )  # (|card_trivial|, n_out)
        W_trivial = -W_trivial / b_trivial[:, None]  # (|card_trivial|, n_in_wo_channel, n_out)

        # consider the least case impact across all outputs
        i_max_trivial: np.ndarray = np.argsort(np.max(W_trivial, 2))  # (|card_trivial|, n_in_wo_channel)
        for j, k in enumerate(card_trivial):
            indices_k: np.ndarray = np.array([i for i in i_max_trivial[j] if i not in xai_indices and i not in free_indices][: cardinality[k]])
            # indices_k = np.array([i for i in range(n_in) if i not in xai_indices and i not in free_indices][:cardinality[k]])
            abstract_free_set[k, indices_k] = 1

    if len(card_index) == 0:
        # only trivial solutions
        return abstract_free_set

    # kept only indices from card_index
    w_u_keep: np.ndarray = w_u[card_index]  # (|card_index|, n_in_with_channel, n_class-1)
    w_u_pos_keep: np.ndarray = w_u_pos[card_index]  # (|card_index|, n_in_with_channel, n_class-1)
    w_u_neg_keep: np.ndarray = w_u_neg[card_index]  # (|card_index|, n_in_with_channel, n_class-1)
    b_u_keep: np.ndarray = b_u[card_index]  # (|card_index|, n_class-1)
    box_keep: np.ndarray = box[card_index]  # (|card_index|, 3, n_in_with_channel)

    W: np.ndarray = get_W(
        w_u_pos_keep, w_u_neg_keep, box_keep, channel=channel, data_format=data_format
    )  # (|card_ind|, n_in_wo_channel, n_out)
    b: np.ndarray = get_b(W, w_u_keep, b_u_keep, box_keep, free_mask)  # (|card_ind|, n_out)

    # set xai and free weights to zero (no impact)
    W = W * (1 - xai_mask) * (1 - free_mask)  # (|card_ind|, n_in_wo_channel, n_out)

    # index_irrelevant:list[int] = [card_index[i] for i in range(len(card_index)) if np.max(b[i]) >0]
    index_knapsack: list[int] = np.array(
        [p for (p, i) in enumerate(card_index) if np.max(b[p]) <= 0]
    )  # (b_g,) b_g <= |card_ind|
    card_knapsack: list[int] = np.array(
        [cardinality[i] for (p, i) in enumerate(card_index) if np.max(b[p]) <= 0]
    )

    if len(index_knapsack) == 0:
        # abstract set of irrelevant features is empty
        return abstract_free_set

    W = W[index_knapsack]  # (b_g, n_in_wo_channel, n_out)
    b = b[index_knapsack]  # (b_g, n_out)

    if method == "milp":
        raise NotImplemented('check the main branch related to the paper')
    elif method == "greedy":
        abstract_free_set = get_greedy(
            card_index[index_knapsack],
            card_knapsack,
            W,
            b,
            xai_indices,
            free_indices,
            abstract_free_set,
        )

    else:
        raise ValueError("method {} is unknown".format(method))

    return abstract_free_set




def free_iteratively_k_features(
    model: Model,
    input_shape: Tuple[int],
    gt_label: int,
    input_sample: np.array,
    lower_bound_input: np.array,
    upper_bound_input: np.array,
    eps: float = 0.0,
    xai_indices: list[int] = [],
    free_indices: list[int] = [],
    data_format: str = "channels_first",
    n_class: int = 10,
    method: str = "greedy",
    lirpa_method:str="CROWN",
    refining_domain: bool = True,
    batch_size_free:int = 10,
    verbose: int = 0,
    step_cardinality = 1, # trying the free every range of features from 1 to n_in_wo_channel with step_cardinality
) -> tuple[list[int], list[int]]:
    """Iteratively finds the largest possible set of robust features for a given input.

    This function implements a high-level iterative algorithm to discover the
    largest set of features that can be perturbed (the "free set") without
    affecting the model's prediction. It provides a formal under-approximation
    of the minimal set of features required to explain a prediction.

    The process involves two main phases:
    1.  **Iterative Set Expansion**: If `refining_domain` is True, it repeatedly
        calls `free_at_once_k_features` to find large groups of features that
        can be safely added to the robust set. After each successful find, it
        expands `free_indices` and repeats the search on the smaller remaining
        pool of features until no more groups can be found.
    2.  **Singleton Refinement**: After the group expansion phase, it performs a
        final, fine-grained search (`free_with_binary_search`) to check if any
        of the remaining individual features can also be safely added to the
        robust set.

    Args:
        model: The model to analyze.
        gt_label: The ground-truth class label.
        input_sample: The nominal input point.
        eps: The radius of the L_inf perturbation.
        xai_indices: A list of feature indices to be explained.
        free_indices: An initial list of features already known to be robust.
        channel: The number of channels in the input data.
        data_format: The data format, "channels_first" or "channels_last".
        n_class: The number of output classes.
        method: The optimization method ("milp" or "greedy") for the sub-problem.
        refining_domain: If True, iteratively expands the robust set until a
            fixed point is reached.
        verbose: Verbosity level.

    Returns:
        A tuple of two lists of feature indices:
        - The first list is the final, largest set of "free" (robust) features found.
        - The second list contains the remaining features that could not be
          proven robust by this method.
    """
    if data_format == "channels_first":
        channel, H, W = input_shape
    else:
        H, W, channel = input_shape

    n_in_with_channel: int = np.prod(input_shape)
    n_in_wo_channel: int = int(n_in_with_channel / channel)


    lower_bound_input: np.ndarray = np.copy(lower_bound_input)
    upper_bound_input: np.ndarray = np.copy(upper_bound_input)

    cardinality: np.ndarray = np.arange(1, n_in_wo_channel-len(free_indices)- len(xai_indices), step_cardinality)


    abstract_set: np.ndarray = free_at_once_k_features(
        model=model,
        input_shape=input_shape,
        gt_label=gt_label,
        input_sample=np.copy(input_sample) + 0.0,
        lower_bound_input=lower_bound_input,
        upper_bound_input=upper_bound_input,
        xai_indices=xai_indices,
        free_indices=free_indices,
        cardinality=cardinality,
        data_format=data_format,
        n_class=n_class,
        method=method,
        lirpa_method=lirpa_method,
        batch_size_free = batch_size_free,
        verbose=verbose,
    )

    card = int(np.max(cardinality))  # we can only free up to the max of features we have successfully freed so far
    if refining_domain:
        while (
            abstract_set.sum(-1).max() != 0 #and card > 1
        ):  # we have found new input features to free among remaining indices
            nb_free = int(abstract_set.sum(-1).max())

            if verbose:
                print('free at once', nb_free)
            i_solution = np.argmax(
                np.sum(abstract_set, -1)
            )  # find the cardinality that propose the largest cardinality to free

            free_indices += [
                i
                for (i, k) in enumerate(abstract_set[i_solution])
                if k == 1 and not i in free_indices
            ]

            # update cardinality: we can only free up to the max of features we have successfully freed so far
            # because recursively we keep expanding the initial abstract domain
            # at initialisation we are not doing np.arange with increment 1, take the consecutive highest value in cardinality
            card = np.where(cardinality > nb_free+1)[0]
            if len(card)==0:
                card=nb_free+step_cardinality
            else:
                card = int(card[0])

            if verbose: 
                print(len(free_indices), 'features freed so far', free_indices, "with cardinality", card)

            if verbose:
                print('next call of free features at once will try to free at most {} features'.format(card))
            cardinality = np.array([i for i in range(1, card)])

            if not len(cardinality):
                raise ValueError(
                    "cardinality should not be empty as the local region is not robust unless the local region is robust"
                )

            abstract_set = free_at_once_k_features(
                model=model,
                input_shape=input_shape,
                gt_label=gt_label,
                input_sample=np.copy(input_sample) + 0.0,
                lower_bound_input=lower_bound_input,
                upper_bound_input=upper_bound_input,
                xai_indices=xai_indices,
                free_indices=free_indices,
                cardinality=cardinality,
                data_format=data_format,
                n_class=n_class,
                method=method,
                lirpa_method=lirpa_method,
                verbose=0,
            )


    # we consider the tightest abstract domain at our disposal: singleton + set of current free features
    # while we find one singleton (we add the one with the least impact according to abstract bound)
    # finish with singleton search
    # warning check lower_bound_input, upper_bound_input

    singleton_free_index: list
    singleton_free_index = free_with_binary_search(
        model=model,
        input_shape = input_shape,
        gt_label=gt_label,
        input_sample=np.copy(input_sample) + 0.0,
        lower_bound=lower_bound_input,
        upper_bound=upper_bound_input,
        free_indices=free_indices,
        potential_candidates=None,  #
        xai_indices=xai_indices,
        channel=channel,
        data_format=data_format,
        n_class=n_class,
        method=lirpa_method,
        verbose=verbose
    )

    return free_indices, singleton_free_index
    
