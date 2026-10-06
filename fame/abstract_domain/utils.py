from typing import Any, List, Tuple, Union

import numpy as np
from fame.abstract_domain.abstract import get_abstract_output_domain

import torch
from torch import Tensor
import numpy as np
from typing import Union, List, Any

def get_upper_box_l0(
    x_min: Tensor,
    x_max: Tensor,
    x_center: Tensor,
    w: Tensor,
    b: Tensor,
    mask_xai: Union[Tensor, np.ndarray],
    mask_free: Union[Tensor, np.ndarray],
    channel: int,
    data_format: str,
    cardinality: Union[int, List[int]],
    **kwargs: Any,
) -> Tensor:
    """Computes the upper bound of a linear operation over a hybrid L-infinity/L0 domain.

    This function implements an abstract transformer for a linear layer (`w*x + b`)
    over a complex perturbation domain. The domain consists of three types of
    features:
    1.  "Free" features (`mask_free`), which are always perturbed within an
        L-infinity box defined by `x_min` and `x_max`.
    2.  "XAI" candidate features (`mask_xai`) which remain at their nominal `x_center` value.
    3. All other features (remaining features) from which up to `cardinality`
        features can be chosen to be perturbed to maximize the output (L0-norm constraint).

    The method works by first calculating the maximum possible positive contribution
    ("score") each feature could make to the output. It then greedily selects
    the top `cardinality` candidate features with the highest scores and sums
    their contributions. The final bound is the sum of the output at the nominal
    center, the contribution from the "free" features, and the contribution from
    the selected top-k "XAI" features.

    Args:
        x_min: Tensor defining the lower bounds of the L-infinity component of the domain.
        x_max: Tensor defining the upper bounds of the L-infinity component of the domain.
        x_center: Tensor for the nominal center of the perturbation domain.
        w: Weight tensor of the linear layer.
        b: Bias tensor of the linear layer.
        mask_xai: A binary mask identifying features that are candidates for L0
            perturbation.
        mask_free: A binary mask identifying features that are always perturbed
            under the L-infinity norm.
        channel: The number of channels in the input data.
        data_format: The data format, either "channels_first" or "channels_last".
        cardinality: The L0 norm budget. The maximum number of features from the
            `xai_mask` pool to perturb. Can be an integer for the entire batch or
            a list of integers for per-sample budgets.
        **kwargs: Additional keyword arguments, used internally.

    Returns:
        A tensor representing the computed upper bound of the linear operation.
    """

    missing_batchsize: bool = "missing_batchsize" in kwargs and kwargs["missing_batchsize"]

    if missing_batchsize:
        w = w.unsqueeze(0)  # Shape: (1, ...)
        b = b.unsqueeze(0)  # Shape: (1, ...)

    # Ensure numpy masks are converted to PyTorch Tensors on the correct device
    device = w.device
    dtype = w.dtype
    
    if isinstance(mask_xai, np.ndarray):
        mask_xai = torch.from_numpy(mask_xai).to(device=device, dtype=dtype)
    else:
        mask_xai = mask_xai.to(device=device, dtype=dtype)

    if isinstance(mask_free, np.ndarray):
        mask_free = torch.from_numpy(mask_free).to(device=device, dtype=dtype)
    else:
        mask_free = mask_free.to(device=device, dtype=dtype)

    # Split into positive and negative components
    n_h: list[int] = list(w.shape[2:])  # output shape dims of the layer
    n_in: int = int(w.shape[1] / channel)  # number of input spatial features

    if data_format == "channels_first":
        w = torch.reshape(w, (-1, channel, n_in) + tuple(n_h))  # Shape: (B, c, n_in, n_h...)
    else:
        w = torch.reshape(w, (-1, n_in, channel) + tuple(n_h))  # Shape: (B, n_in, c, n_h...)

    w_pos: Tensor = torch.relu(w)  # Shape: (B, c, n_in, n_h...)
    w_neg: Tensor = w - w_pos       # Shape: (B, c, n_in, n_h...)

    axis_channel: int
    if data_format == "channels_first":
        x_min_out: Tensor = torch.reshape(x_min, [-1, channel, n_in] + [1] * len(n_h))  # Shape: (B, c, n_in, 1...)
        x_max_out: Tensor = torch.reshape(x_max, [-1, channel, n_in] + [1] * len(n_h))
        x_center_out: Tensor = torch.reshape(x_center, [-1, channel, n_in] + [1] * len(n_h))
        axis_channel = 1
    else:
        x_min_out: Tensor = torch.reshape(x_min, [-1, n_in, channel] + [1] * len(n_h))  # Shape: (B, n_in, c, 1...)
        x_max_out: Tensor = torch.reshape(x_max, [-1, n_in, channel] + [1] * len(n_h))
        x_center_out: Tensor = torch.reshape(x_center, [-1, n_in, channel] + [1] * len(n_h))
        axis_channel = 2

    mask_xai_out: Tensor = torch.reshape(mask_xai, [-1, n_in] + [1] * len(n_h))   # Shape: (B, n_in, 1...)
    mask_free_out: Tensor = torch.reshape(mask_free, [-1, n_in] + [1] * len(n_h))

    # Sum positive contributions along the channel dimension
    scoring_samples: Tensor = torch.sum(
        w_pos * (x_max_out - x_center_out) + w_neg * (x_min_out - x_center_out), 
        dim=axis_channel
    )  # Shape: (B, n_in, n_h...)

    # Exclude xai and free masked features from candidate top-K selection
    scoring_samples_wo_free: Tensor = (
        scoring_samples * (1.0 - mask_xai_out) * (1.0 - mask_free_out)
    )  # Shape: (B, n_in, n_h...)

    # Select threshold using PyTorch sorting (descending order)
    if isinstance(cardinality, int):
        # Sort in descending order along axis 1 (n_in)
        sorted_scores, _ = torch.sort(scoring_samples_wo_free, dim=1, descending=True)  # Shape: (B, n_in, n_h...)
        threshold: Tensor = sorted_scores[:, cardinality - 1:cardinality]               # Shape: (B, 1, n_h...)
    else:
        batch_size: int = len(cardinality)
        sorted_scores, _ = torch.sort(scoring_samples_wo_free, dim=1, descending=True)
        
        # Batch indexing for variable per-sample cardinalities
        cardinality_tensor = torch.tensor(cardinality, device=device) - 1
        threshold: Tensor = sorted_scores[torch.arange(batch_size, device=device), cardinality_tensor].unsqueeze(1) # Shape: (B, 1, n_h...)

    # Mask to keep only scoring_samples greater than or equal to threshold
    final_score_mask: Tensor = (scoring_samples_wo_free >= threshold).to(dtype=dtype)
    final_score: Tensor = torch.sum(
        scoring_samples_wo_free * final_score_mask, dim=1
    )  # Shape: (B, n_h...)

    # Calculate nominal center bias + sum over channel and spatial input dims
    bias: Tensor = b + torch.sum(
        torch.sum(w * x_center_out, dim=axis_channel), dim=1
    )  # Shape: (B, n_h...)

    # Add free mask contributions directly
    free_scoring_samples: Tensor = torch.sum(scoring_samples * mask_free_out, dim=1)

    bias = bias + free_scoring_samples
    return final_score + bias


def get_lower_box_l0(
    x_min: Tensor,
    x_max: Tensor,
    x_center: Tensor,
    w: Tensor,
    b: Tensor,
    mask_xai: np.ndarray,
    mask_free: np.ndarray,
    channel: int,
    data_format: str,
    cardinality: Union[int, List[int]],
    **kwargs: Any,
) -> Tensor:
    """Computes the lower bound of a linear operation over a hybrid L-infinity/L0 domain.

    This function is the counterpart to `get_upper_box_l0`. It computes the tightest
    possible lower bound for a linear operation (`w*x + b`) over the same complex
    perturbation domain, which combines L-infinity and L0-norm constraints.

    The implementation leverages the duality principle that the minimum of a function
    is the negative of the maximum of its negative, i.e.,
    $min(f(x)) = -max(-f(x))$.
    It computes the lower bound by calling `get_upper_box_l0` with negated
    weights (`-w`) and bias (`-b`) and then negating the result. This effectively
    finds the perturbation that minimizes the linear function's output.

    Args:
        x_min: Tensor defining the lower bounds of the L-infinity component of the domain.
        x_max: Tensor defining the upper bounds of the L-infinity component of the domain.
        x_center: Tensor for the nominal center of the perturbation domain.
        w: Weight tensor of the linear layer.
        b: Bias tensor of the linear layer.
        mask_xai: A binary mask identifying features that are set to their nominal value
        mask_free: A binary mask identifying features that are always perturbed
            under the L-infinity norm.
        channel: The number of channels in the input data.
        data_format: The data format, either "channels_first" or "channels_last".
        cardinality: The L0 norm budget. The maximum number of features from the
            `xai_mask` pool to perturb. Can be an integer for the entire batch or
            a list of integers for per-sample budgets.
        **kwargs: Additional keyword arguments, passed to the underlying
            `get_upper_box_l0` function.

    Returns:
        A tensor representing the computed lower bound of the linear operation.
    """
    return -get_upper_box_l0(
        x_min=x_min,
        x_max=x_max,
        x_center=x_center,
        w=-w,
        b=-b,
        mask_xai=mask_xai,
        mask_free=mask_free,
        channel=channel,
        data_format=data_format,
        cardinality=cardinality,
        **kwargs,
    )


def check_is_robust(
    model, input_sample, eps, channel, data_format, n_class, lirpa_model=None, means=None, stddev=None
) -> bool:
    """Checks the L-infinity robustness of a model for a given input and epsilon.

    This function verifies whether the model's prediction for a given `input_sample`
    remains constant within an $L_\infty$ ball of radius `eps`. The perturbation
    space is clipped to the valid data range of [0, 1].

    It uses abstract interpretation via the `get_abstract_output_domain` function
    to compute a sound upper bound on the logit differences ($z_{gt} - z_{other}$)
    over the entire input perturbation region, where $z_{gt}$ is the logit of the
    predicted class for the original `input_sample`.

    If the maximum of these upper bounds is less than or equal to zero, it
    formally proves that the original prediction is robust for any input within
    the specified $L_\infty$ ball.

    Args:
        model: The pytorch model to verify.
        input_sample: A single input point (e.g., an image) around which
            robustness is checked.
        eps: The radius (epsilon) of the $L_\infty$ norm perturbation.
        channel: The number of channels in the input data.
        data_format: The data format, either "channels_first" or "channels_last".
        n_class: The number of output classes of the model.
        lirpa_model: An optional, pre-compiled LiRPA model for improved
            performance.

    Returns:
        `True` if the model is provably robust for the given input and
        epsilon, `False` otherwise.
    """

    n_in_wo_channel: int = int(input_sample.shape[-1] / channel)
    free_indices: list[int] = [i for i in range(n_in_wo_channel)]

    if means is None and stddev is None:
        lower_bound = np.maximum(input_sample - eps, 0.0)
        upper_bound = np.minimum(input_sample + eps, 1.0)
    else:
        lower_bound = np.maximum(input_sample - eps, - (means/stddev))
        upper_bound = np.minimum(input_sample + eps, ((1-means)/stddev) )
        
    upper: np.array = get_abstract_output_domain(
        lirpa_model=model,
        input_sample=input_sample,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        free_indices=free_indices,
        channel=channel,
        data_format=data_format,
        n_class=n_class,
    )  # (1, n_out)

    return np.max(upper) <= 0
