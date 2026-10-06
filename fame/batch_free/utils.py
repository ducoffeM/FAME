import numpy as np


def encode_matrix(n_class: int, groundtruth: int) -> np.ndarray:
    """Creates the property matrix `C` for robustness verification.

    This matrix is used in backward bound propagation methods (like CROWN) to
    certify properties of the form $z_i - z_{gt} \\le 0$, where $z_{gt}$ is the
    logit of the ground-truth class.

    The resulting matrix has a shape of `(n_class, n_class - 1)`. When multiplied
    with a logit vector, it produces a vector of logit differences against the
    ground-truth logit.

    Args:
        n_class: The total number of output classes.
        groundtruth: The index of the ground-truth label.

    Returns:
        The property matrix `C` of shape `(n_class, n_class - 1)`.
    """
    gt_vec = np.zeros((n_class,))
    gt_vec[groundtruth] = 1
    C_ = np.diag(np.ones((n_class,))) - gt_vec
    return C_[[i for i in range(n_class) if i != groundtruth]].T


## Abstract Free Set
def get_b(
    W: np.ndarray, w_u: np.ndarray, b_u: np.ndarray, box: np.ndarray, free_mask: np.ndarray
) -> np.ndarray:
    """Computes the constant bias term of an affine bound for a robustness problem.

    This function calculates the effective bias term `b` for an affine upper bound
    of the form `upper_bound = sum(W_i * x_i) + b`. This bias is composed of two parts:
    1. The output of the original affine function (`w_u*x + b_u`) evaluated at the
       nominal center of the input domain (`box`).
    2. The maximum possible contribution from all features that are designated as
       "free" (i.e., always perturbed), whose impacts are given by `W`.

    Args:
        W: The feature-wise impact coefficients, from `get_W`.
        w_u: The weights of the original affine upper bound from a lirpa model.
        b_u: The bias of the original affine upper bound from a lirpa model.
        box: The input domain tensor `[lower, upper, center]`.
        free_mask: A binary mask identifying the "free" features.

    Returns:
        A numpy array `b` of shape `(batch, n_outputs)` representing the final bias term.
    """
    # W (batch_size, n_in_wo_channel, n_out)
    # w_u (batch_size, n_in_with_channel, n_out)
    # b_u (batch_size, n_out)
    # box (batch_size, 3, n_in_with_channel)
    # free mask (1, n_in_wo_channel, 1)
    center_: np.ndarray = box[:, 2, :, None]  # (batch_size, n_in_with_channel, 1)

    b: np.ndarray = np.sum(w_u * center_, 1) + b_u  # (batch_size, n_out)
    # add irrelevant features
    b = b + np.sum(W * free_mask, 1)  # (batch_size, n_out)

    return b


def get_W(
    w_u_pos: np.ndarray,
    w_u_neg: np.ndarray,
    box: np.ndarray,
    channel: int = 1,
    data_format: str = "channels_first",
) -> np.ndarray:
    """Computes the feature-wise impact coefficients (`W`) from an affine relaxation.

    This function takes the weights of an affine relaxation (`w_u`) and an input
    domain (`box`) to compute the maximum possible contribution of each input
    feature to the final upper bound. The resulting `W` matrix represents these
    per-feature "scores" or "impacts," which are essential for formulating the
    robustness problem as a knapsack-style optimization.

    Args:
        w_u_pos: The positive part of the affine weights (`relu(w_u)`).
        w_u_neg: The negative part of the affine weights (`w_u - relu(w_u)`).
        box: The input domain tensor `[lower, upper, center]`.
        channel: The number of channels in the input data.
        data_format: The data format, either "channels_first" or "channels_last".

    Returns:
        A numpy array `W` of shape `(batch, n_features, n_outputs)` containing
        the impact coefficients for each feature.
    """
    n_in_with_channel: int = box.shape[-1]
    n_in_wo_channel: int = int(n_in_with_channel / channel)
    batch_size: int = w_u_pos.shape[0]

    # w_u_(pos, neg) (batch_size, n_in_with_channel, n_out)
    lower_: np.ndarray = box[:, 0, :, None]  # (batch_size, n_in_with_channel, 1)
    upper_: np.ndarray = box[:, 1, :, None]  # (batch_size, n_in_with_channel, 1)
    center_: np.ndarray = box[:, 2, :, None]  # (batch_size, n_in_with_channel, 1)

    # rescale coefficient and bias
    w: np.ndarray = w_u_pos * (upper_ - center_) + w_u_neg * (
        lower_ - center_
    )  # (batch_size, n_in_with_channel, n_out)
    # reshape channel dimension and sum over it
    if data_format == "channels_first":
        w = np.sum(
            np.reshape(w, (batch_size, channel, n_in_wo_channel, -1)), 1
        )  # (batch_size, n_in_wo_channel, n_out)
    else:
        w = np.sum(
            np.reshape(w, (batch_size, n_in_wo_channel, channel, -1)), 2
        )  # (batch_size, n_in_wo_channel, n_out)

    return w


def get_xai_mask(n_in: int, xai_indices: list[int]) -> np.ndarray:
    """Creates a broadcastable binary mask from a list of XAI feature indices.

    Args:
        n_in: The total number of features (without channels).
        xai_indices: A list of indices to be marked as 1 in the mask.

    Returns:
        A numpy array of shape `(1, n_in, 1)` representing the binary mask,
        ready for broadcasting.
    """
    return get_mask(n_in, xai_indices)


def get_free_mask(n_in: int, free_indices: list[int]) -> np.ndarray:
    """Creates a broadcastable binary mask from a list of free feature indices.

    Args:
        n_in: The total number of features (without channels).
        free_indices: A list of indices to be marked as 1 in the mask.

    Returns:
        A numpy array of shape `(1, n_in, 1)` representing the binary mask,
        ready for broadcasting.
    """
    return get_mask(n_in, free_indices)

def get_mask(n_in: int, indices: list[int]) -> np.ndarray:
    """Creates a broadcastable binary mask from a list of feature indices.

    Args:
        n_in: The total number of features (without channels).
        indices: A list of indices to be marked as 1 in the mask.

    Returns:
        A numpy array of shape `(1, n_in, 1)` representing the binary mask,
        ready for broadcasting.
    """
    mask: np.ndarray = np.zeros((n_in,))
    mask[indices] = 1  # 1 if potential candidate to add in the features
    mask = mask[None, :, None]  # (1, n_in, 1)
    return mask

