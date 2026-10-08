import numpy as np
import torch
import torch.nn as nn

from fame.attack import find_closest_xai
from fame.batch_free.free import free_at_once_k_features, free_iteratively_k_features



class Reshape(nn.Module):

    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)


def get_xai_fame(model: torch.nn.Module, input_sample: np.ndarray, input_shape: tuple, gt_label: int,n_class: int, \
                eps: float, lower_bound_input: np.ndarray = None, upper_bound_input: np.ndarray = None, \
                data_format: str = "channels_first", free_methods=["CROWN"], xai_methods=['pgd'], traversal_order="lirpa", verbose:int=0):
    """
    Get the FAME XAI indices for a given model and input sample.

    Args:
        model: The neural network model.
        input_sample: The input sample for which to compute the XAI indices size (C, H, W).
        input_shape: The shape of the input sample.
        gt_label: The ground truth label for the input sample.
        n_class: The number of classes in the model's output.
        eps: The epsilon value for the FAME algorithm.
        lower_bound_input: The lower bound of the input sample size (C,H,W).
        upper_bound_input: The upper bound of the input sample size (C,H,W).
        data_format: The format of the input data.
        free_methods: The methods to use for finding free indices.
        xai_methods: The methods to use for finding XAI indices.
        traversal_order: the ordering use to select XAI indices.
        verbose: The level of verbosity for the FAME algorithm.

    Returns:
        xai_indices: The indices of the features that are considered important by the FAME algorithm.
        free_indices: The indices of the features that are considered free by the FAME algorithm.
        remaining_indices: The indices of the features that are not considered important or free by the FAME algorithm.
    """


    if data_format=="channels_first":
        channel, H, W = input_shape
    else:
        raise NotImplementedError("auto lirpa does not support channel_last")

    device = next(model.parameters()).device

    flatten_image = np.reshape(input_sample, (channel,H*W))
    flatten_model =  nn.Sequential(Reshape(-1, *input_shape), model)
    flatten_model.eval()
    flatten_model.to(next(model.parameters()).device)


    flatten_lower_bound = np.reshape(lower_bound_input, (channel,H*W)) if lower_bound_input is not None else flatten_image - eps
    flatten_upper_bound = np.reshape(upper_bound_input, (channel,H*W)) if upper_bound_input is not None else flatten_image + eps

    free_indices:list = []
    xai_indices:list = []

    for free_method in free_methods:

        if verbose:
            print("free method:", free_method)
        free_indices_cardinality, free_indices_binary = free_iteratively_k_features(
                model=model,
                input_shape=input_shape,
                gt_label=gt_label,
                input_sample=np.copy(flatten_image),
                lower_bound_input=np.copy(flatten_lower_bound),
                upper_bound_input=np.copy(flatten_upper_bound),
                data_format= data_format,
                eps=eps,
                n_class = n_class,
                method=free_method,
                verbose=verbose,
            )

        free_indices:list = free_indices_cardinality+free_indices_binary


    extra_free=[] # samples that are used to create attacks, thus we cannot put them in xai
    for xai_method in xai_methods:

        if verbose:
            print("xai_method: ", xai_method)

        potential_xai, unknown = find_closest_xai(
                    model=flatten_model,
                    gt_label=gt_label,
                    input_sample=np.reshape(flatten_image, (-1,)),
                    lower_bound=np.reshape(flatten_lower_bound, (-1,)),
                    upper_bound=np.reshape(flatten_upper_bound, (-1,)),
                    eps=eps,
                    xai_indices=xai_indices,
                    free_indices=free_indices,
                    method=xai_method,
                    device=device,
                    channel=channel,
                    data_format=data_format,
                    n_class=n_class,
                    traversal_order="lirpa",
                    verbose=verbose,
            )
        xai_indices:list = list(set(potential_xai + xai_indices))

        if verbose:
            print('xai', len(xai_indices))

        

        assert set(potential_xai).isdisjoint(set(free_indices)), "xai should not overlap with free_indices"

    remaining_indices = [ i for i in range(W*H) if i not in xai_indices and i not in free_indices ]

    return xai_indices, free_indices, remaining_indices
