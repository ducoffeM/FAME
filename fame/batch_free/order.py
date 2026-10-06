import torch
import numpy as np
from fame.batch_free.utils import (
    encode_matrix,
    get_b,
    get_free_mask,
    get_W,
    get_xai_mask,
)

from ..abstract_domain.abstract import get_abstract_model

from auto_LiRPA.perturbations import Perturbation, PerturbationLpNorm
from auto_LiRPA import BoundedModule, BoundedTensor



def get_trivial(
    w_u_trivial: np.ndarray,
    b_u_trivial: np.ndarray,
    box_trivial: np.ndarray,
    xai_mask=np.ndarray,
    free_mask=np.ndarray,
    channel: int = 1,
    data_format: str = "channels_first",
):
    """Calculates the normalized impact scores for features from an affine relaxation.

    This helper function takes the affine parameters (`w_u`, `b_u`) of a network's
    upper bound, obtained from abstract interpretation, and computes a normalized
    impact score for each input feature.

    This score represents the "efficiency" of a feature in worsening the robustness
    bound, analogous to a value-to-weight ratio in knapsack problems. It is
    calculated by determining the maximum possible influence of each feature on the
    bound (`get_W`) and then normalizing it by the bias term (`b_trivial`).

    Args:
        w_u_trivial: The weights of the affine upper bound from a decomon model.
        b_u_trivial: The bias of the affine upper bound from a decomon model.
        box_trivial: The input domain tensor `[lower, upper, center]` that was used
            for the abstract analysis.
        xai_mask: A binary mask to exclude pre-defined XAI features from the
            score calculation.
        free_mask: A binary mask to exclude pre-defined free features from the
            score calculation.
        channel: The number of channels in the input data.
        data_format: The data format, either "channels_first" or "channels_last".

    Returns:
        A numpy array containing the normalized impact scores for each feature, which
        can be used for ranking or greedy selection.
    """
    # we could return all indices up to cardinalities
    # best is to return the highest one from the abstract domain (to facilitate freeing latter one)

    # we keep only indices from card_trivial
    w_u_pos_trivial: np.ndarray = np.maximum(w_u_trivial, 0.0)  # (|card_trivial|, n_in, n_out)
    w_u_neg_trivial: np.ndarray = np.minimum(w_u_trivial, 0.0)  # (|card_trivial|, n_in, n_out)

    W_trivial: np.ndarray = get_W(
        w_u_pos_trivial, w_u_neg_trivial, box_trivial, channel=channel, data_format=data_format
    )  # (|card_trivial|, n_in, n_out)
    # set xai and free weights to zero (no impact)
    W_trivial = W_trivial * (1 - xai_mask) * (1 - free_mask)
    b_trivial: np.ndarray = get_b(
        W_trivial, w_u_trivial, b_u_trivial, box_trivial, free_mask
    )  # (|card_trivial|, n_out)
    W_trivial = -W_trivial / b_trivial[:, None]  # (|card_trivial|, 1, n_out)

    return W_trivial


def get_greedy_order(
    model: torch.nn.Module,
    input_sample: np.ndarray,
    gt_label: int,
    lower_bound,
    upper_bound: np.ndarray,
    free_indices: list[int],
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
):
    """Ranks input features by their impact on model robustness to get a greedy search order.

    This function provides an ordered list of features, ranked from most to least
    influential, for use in greedy search algorithms. A feature's influence is
    determined by its potential to adversely affect the model's certified
    robustness bounds, as measured by abstract interpretation.

    The process is as follows:
    1.  Perform an abstract analysis using `autolirpa` to obtain a sound affine
        approximation (`w*x + b`) of the network's output bounds.
    2.  Use this approximation to calculate a normalized impact score for each
        feature via the `get_trivial` helper function.
    3.  Sort the features in descending order based on these scores.

    Args:
        model: The pytorch model to be analyzed.
        input_sample: The nominal input point.
        gt_label: The ground-truth class label.
        lower_bound: The lower bounds of the L-infinity perturbation.
        upper_bound: The upper bounds of the L-infinity perturbation.
        xai_indices: A list of feature indices to be excluded from the ordering.
        free_indices: A list of feature indices to be excluded from the ordering.
        channel: The number of channels in the input data.
        data_format: The data format, either "channels_first" or "channels_last".
        n_class: The number of output classes of the model.

    Returns:
        A list of feature indices, sorted from most to least impactful, to be
        used as a traversal order in greedy algorithms.
    """
    # we should either find it is safe using abstract interpretation or not find any attacks
    device = next(model.parameters()).device
    xai_indices=[]

    n_in_with_channel: int = input_sample.shape[-1] # working with a flatten model
    n_in_wo_channel: int = int(n_in_with_channel / channel)
    # freeze xai features to the nominal value
    if len(xai_indices):
        lower_bound_c: np.ndarray
        upper_bound_c: np.ndarray
        input_sample_c: np.ndarray

        if data_format == "channels_first":
            lower_bound_c = np.reshape(lower_bound, (channel, n_in_wo_channel))
            upper_bound_c = np.reshape(upper_bound, (channel, n_in_wo_channel))
            input_sample_c = np.reshape(input_sample, (channel, n_in_wo_channel))
            lower_bound_c[:, xai_indices] = input_sample_c[:, xai_indices]
            upper_bound_c[:, xai_indices] = input_sample_c[:, xai_indices]

        else:
            lower_bound_c = np.reshape(lower_bound, (channel, n_in_wo_channel))
            upper_bound_c = np.reshape(upper_bound, (channel, n_in_wo_channel))
            input_sample_c = np.reshape(input_sample, (channel, n_in_wo_channel))
            lower_bound_c[:, xai_indices] = input_sample_c[:, xai_indices]
            upper_bound_c[:, xai_indices] = input_sample_c[:, xai_indices]

        lower_bound = np.reshape(lower_bound_c, (n_in_with_channel,))
        upper_bound = np.reshape(upper_bound_c, (n_in_with_channel,))

    

    box: np.ndarray = np.concatenate(
        [lower_bound[None, None], upper_bound[None, None]], 1
    )  # (1, 3, n_in)

    eps:float = np.max(upper_bound - lower_bound)
    input_sample_tensor:Tensor = torch.tensor(input_sample[None]).to(device)
    lower_bound_tensor:Tensor = torch.tensor(lower_bound[None]).to(device)
    upper_bound_tensor:Tensor = torch.tensor(upper_bound[None]).to(device)

    # build your input domain
    C_gt: np.ndarray = encode_matrix(n_class=10, groundtruth=gt_label)[None]  # (1, n_class, n_class -1)
    C_gt = np.transpose(C_gt, (0, 2, 1))  # (current_batch_size, n_class-1, n_class)
    # convert to torch tensor
    C_gt: Tensor = torch.tensor(C_gt, dtype=torch.float32).to(device)

    lirpa_model = get_abstract_model(model=model,dummy_input=input_sample_tensor).to(device)
    ptb:Perturbation = PerturbationLpNorm(norm=np.inf, eps=eps,
                                                x_L = lower_bound_tensor, x_U = upper_bound_tensor)
    bounded_image = BoundedTensor(input_sample_tensor, ptb)



    input_node_name = lirpa_model.input_name[0]
    # 2. Specify that we need the A matrices for output node with respect to input node
    needed_A_dict = {lirpa_model.output_name[0]: [input_node_name]}    

    _, ub_output, A_dict = lirpa_model.compute_bounds(x=(bounded_image,),method="CROWN", \
                                                        return_A=True, needed_A_dict=needed_A_dict, C=C_gt,\
                                                        bound_lower=False, bound_upper=True)
    A_info = A_dict[lirpa_model.output_name[0]][input_node_name]

    A_upper = A_info['uA']     # Upper affine slope matrix,
    b_upper = A_info['ubias']  # Upper affine bias,
        
    # detach and cast to numpy arrays

    w_u = A_upper.detach().cpu().numpy()
    b_u = b_upper.detach().cpu().numpy()    

    # reshape w_u to (batch, n_in_with_channel, n_class-1)
    w_u = np.reshape(w_u, (w_u.shape[0], -1, b_u.shape[1]))


    xai_mask: np.ndarray = get_xai_mask(n_in_wo_channel, xai_indices)  # (1, n_in, 1)
    free_mask: np.ndarray = get_free_mask(n_in_wo_channel, free_indices)
    box = np.concatenate([box, input_sample[None, None]], axis=1)
    # get trivial coefficients
    W_trivial = get_trivial(
        w_u_trivial=w_u,
        b_u_trivial=b_u,
        box_trivial=box,
        xai_mask=xai_mask,
        free_mask=free_mask,
        channel=channel,
        data_format=data_format,
    )  # (1, n_in_wo_channel, n_out)

    # upper (1, n_out)
    potential_xai: list[int] = [
        i for i in np.argsort(-np.max(W_trivial[0], -1), -1) if i not in xai_indices + free_indices
    ]
    return potential_xai
