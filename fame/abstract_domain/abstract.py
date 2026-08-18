from typing import Tuple, Union

import keras
import numpy as np
from decomon import clone
from keras import KerasTensor as Tensor
from keras.layers import Input

from ..batch_free.utils import encode_matrix


def get_abstract_model(
    model: keras.models.Model, n_class: int = 10, final_affine: bool = False, final_ibp: bool = True
) -> keras.models.Model:
    """Create a decomon model that computes abstract upper bounds for robustness verification.

    This function takes a standard Keras model and converts it into a decomon model
    using `decomon.models.clone`. The resulting model is designed to compute the
    upper bounds of linear combinations of the original model's outputs (logits),
    which is a key step in formal verification and robustness certification.

    The new model introduces an additional input tensor `C` for backward bound
    propagation (e.g., CROWN). This `C` tensor specifies the properties to be
    verified, typically the difference between logits $z_i - z_j$ for all $j \\neq i$.
    The output of the returned model will be the upper bounds for these differences.
    If an output value is negative, it proves that the corresponding property holds
    (e.g., $z_i < z_j$).

    Args:
        model: The input Keras model to be converted.
        n_class: The number of output classes of the model. Defaults to 10.
        final_affine: Whether the relaxation of the final layer is affine.
            Defaults to False.
        final_ibp: Specifies whether to resolve the symbolic affine bounds of the final layer into a concrete numerical interval ([lower, upper]). Defaults to True

    Returns:
        A new decomon model that takes the original inputs plus an
        additional input tensor `C` of shape `(n_class, n_class - 1)`.
        The model's output represents the upper bounds of the properties
        defined by `C`.
    """

    C: Tensor = Input((n_class, n_class - 1))
    # (batch_size, n_class, n_class-1)

    decomon_model: keras.model.Model = clone(
        model,
        final_affine=final_affine,
        final_ibp=final_ibp,
        final_lower=False,
        backward_bounds=[C],
    )  # return only upper bound

    return decomon_model


def get_abstract_output_domain_singleton(
    model: keras.models.Model,
    input_sample: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    xai_indices: list[int],
    free_indices: list[int],
    remaining_indices: Union[list[int], None] = None,  # a subset of remaining dimensions or None
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    decomon_model: keras.models.Model = None,
    batch_size: int =25,
) -> np.ndarray:
    """Computes the certified output bounds when perturbing single features one by one.

    This function performs an abstract analysis to determine the impact of
    perturbing individual input features. It constructs a batch of input domains
    (hyper-rectangles) to be passed to a decomon model.

    In each domain of the batch:
    - Features in `xai_indices` are fixed to their nominal values from `input_sample`.
    - Features in `free_indices` are always allowed to vary within their global
      `lower_bound` and `upper_bound`.
    - Exactly one feature from `remaining_indices` is allowed to vary within
      its global bounds.

    This setup allows for efficiently calculating the certified effect of each
    "remaining" feature in isolation, while other features are either fixed or
    also perturbed. The function returns the upper bounds on the logit
    differences $z_{gt} - z_{other}$, where $z_{gt}$ is the logit of the
    ground-truth class.

    Args:
        model: The original Keras neural network model.
        input_sample: A single nominal input point (e.g., an image).
        lower_bound: An array defining the lower bounds for the entire
            perturbation space.
        upper_bound: An array defining the upper bounds for the entire
            perturbation space.
        xai_indices: A list of feature indices that will be **fixed** to their
            nominal `input_sample` value.
        free_indices: A list of feature indices that will **always be perturbed**
            within their bounds in the background.
        remaining_indices: A list of feature indices to be analyzed one by one.
            Each feature in this list will be perturbed in its own domain
            within the batch. If `None`, it defaults to all features not in
            `xai_indices` or `free_indices`.
        channel: The number of channels in the input data. Defaults to 1.
        data_format: The data format, either "channels_first" or "channels_last".
            Defaults to "channels_first".
        n_class: The number of output classes of the model. Defaults to 10.
        decomon_model: An optional, pre-compiled decomon model for efficiency.
            If not provided, it will be created internally.

    Returns:
        A numpy array of shape `(len(remaining_indices), n_class - 1)`. Each
        row `i` contains the certified upper bounds on the logit differences
        for the input domain where the `i`-th feature from `remaining_indices`
        was perturbed. A negative value proves robustness for that check.
    """

    n_in_with_channel: int = input_sample.shape[-1]
    n_in_wo_channel: int = int(n_in_with_channel / channel)

    if data_format == "channels_first":
        # (channel, n_in_wo_channel)
        input_sample_c: np.ndarray = np.reshape(input_sample, (channel, n_in_wo_channel))
        lower_bound_c: np.ndarray = np.reshape(lower_bound, (channel, n_in_wo_channel))
        upper_bound_c: np.ndarray = np.reshape(upper_bound, (channel, n_in_wo_channel))
    else:  # channels_last
        # (n_in_wo_channel, channel)
        input_sample_c: np.ndarray = np.reshape(input_sample, (n_in_wo_channel, channel))
        lower_bound_c: np.ndarray = np.reshape(lower_bound, (n_in_wo_channel, channel))
        upper_bound_c: np.ndarray = np.reshape(upper_bound, (n_in_wo_channel, channel))

    max_batch_size: int
    if remaining_indices is None:
        # consider every remaining dimensions
        max_batch_size = (
            n_in_wo_channel - len(xai_indices) - len(free_indices)
        )  # number of remaining indices
    else:
        # remaining_indices has been set previously by the pipeline
        max_batch_size = len(remaining_indices)

    if (batch_size < 0) or (batch_size > max_batch_size):
        batch_size = max_batch_size

    gt_label: int = model.predict(input_sample[None], verbose=0).argmax(-1)[0]

    # repeat subdomains
    # freeze xai features to the nominal value
    lower_bound_np: np.ndarray = (
        np.copy(input_sample_c) + 0.0
    )  # (channel, n_in_wo_channel) or (n_in_wo_channel, channel)
    upper_bound_np: np.ndarray = np.copy(input_sample_c) + 0.0
    if len(free_indices):
        if data_format == "channels_first":
            lower_bound_np[:, free_indices] = lower_bound_c[
                :, free_indices
            ]  # open the free dimension
            upper_bound_np[:, free_indices] = upper_bound_c[:, free_indices]
        else:
            lower_bound_np[free_indices, :] = lower_bound_c[
                free_indices, :
            ]  # open the free dimension
            upper_bound_np[free_indices, :] = upper_bound_c[free_indices, :]

    # expand one dimension in the set of remaining features
    if remaining_indices is None:
        remaining_indices = [
            i for i in range(n_in_wo_channel) if i not in xai_indices + free_indices
        ]

    assert (
        len(remaining_indices) == max_batch_size
    ), "Value Error remaining indices length should match batch_size"

    all_upper = []
    for k in range(0, max_batch_size, batch_size):
        # repeat
        k_stop = min(k + batch_size, max_batch_size)
        current_batch_size = k_stop - k
        lower_bound_batch: np.ndarray = np.repeat(
            lower_bound_np[None], repeats=current_batch_size, axis=0
        )  # (batch_size, channel, n_in_wo_channel)
        upper_bound_batch: np.ndarray = np.repeat(
            upper_bound_np[None], repeats=current_batch_size, axis=0
        )  # (batch_size, channel, n_in_wo_channel)


        for i, j in enumerate(remaining_indices[k:k_stop]):
            if data_format == "channels_first":
                lower_bound_batch[i, :, j] = lower_bound_c[:, j]
                upper_bound_batch[i, :, j] = upper_bound_c[:, j]
            else:
                try:
                    lower_bound_batch[i, j, :] = lower_bound_c[j, :]
                    upper_bound_batch[i, j, :] = upper_bound_c[j, :]
                except:
                    import pdb

                    pdb.set_trace()

        # flatten lower_bound_batch and upper_bound_batch
        lower_bound_batch = np.reshape(
            lower_bound_batch, (-1, 1, n_in_with_channel)
        )  # (batch_size, 1, n_in_with_channel)
        upper_bound_batch = np.reshape(
            upper_bound_batch, (-1, 1, n_in_with_channel)
        )  # (batch_size, 1, n_in_with_channel)
        box: np.ndarray = np.concatenate(
            [lower_bound_batch, upper_bound_batch], 1
        )  # (batch_size, 2, n_in_with_channel)

        # build your input domain
        # encode matrix C
        C_gt: np.ndarray = np.repeat(
            encode_matrix(n_class=n_class, groundtruth=gt_label)[None], repeats=current_batch_size, axis=0
        )
        # (batch_size, n_class, n_class-1)

        if decomon_model is None:
            C: Tensor = Input((n_class, n_class - 1))

            decomon_model: keras.model.Model = clone(
                model, final_affine=False, final_ibp=True, final_lower=False, backward_bounds=[C]
            )  # return only upper bound

        upper: np.ndarray = decomon_model.predict([box, C_gt], verbose=0)
        all_upper.append(upper)

    upper = np.concatenate(all_upper, axis=0)
    return upper


def get_abstract_output_domain(
    model: keras.models.Model,
    input_sample: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    free_indices: list[int],
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    decomon_model: keras.models.Model = None,
) -> np.ndarray:
    """Computes the certified output bounds for a single input domain.

    This function performs abstract interpretation on a single hyper-rectangular
    input domain to get a certified guarantee on the model's output. The domain
    is constructed by allowing a specific subset of features to vary while
    keeping others fixed.

    The input domain is defined as follows:
    - Features in `free_indices` are allowed to vary within their global
      `lower_bound` and `upper_bound`.
    - All other features are **fixed** to
      their nominal values from `input_sample`.

    It then uses a decomon model to compute the certified upper bounds on the
    logit differences, $z_{gt} - z_{other}$, where $z_{gt}$ is the logit of the
    ground-truth class for the nominal `input_sample`.

    Args:
        model: The original Keras neural network model.
        input_sample: A single nominal input point used as the reference for
            fixing feature values.
        lower_bound: An array defining the lower bounds for the entire
            perturbation space.
        upper_bound: An array defining the upper bounds for the entire
            perturbation space.
        free_indices: A list of feature indices that will be **allowed to vary**
            within their bounds for this analysis.
        channel: The number of channels in the input data. Defaults to 1.
        data_format: The data format, either "channels_first" or "channels_last".
            Defaults to "channels_first".
        n_class: The number of output classes of the model. Defaults to 10.
        decomon_model: An optional, pre-compiled decomon model for efficiency.
            If not provided, it will be created internally.

    Returns:
        A numpy array of shape `(1, n_class - 1)`. It contains the certified
        upper bounds on the logit differences. A negative value proves that the
        ground-truth class is robustly the maximum for any input within the
        defined domain.
    """

    n_in_with_channel: int = input_sample.shape[-1]
    n_in_wo_channel: int = int(n_in_with_channel / channel)

    if data_format == "channels_first":
        # (channel, n_in_wo_channel)
        input_sample_c: np.ndarray = np.reshape(input_sample, (channel, n_in_wo_channel))
        lower_bound_c: np.ndarray = np.reshape(lower_bound, (channel, n_in_wo_channel))
        upper_bound_c: np.ndarray = np.reshape(upper_bound, (channel, n_in_wo_channel))
    else:  # channels_last
        # (n_in_wo_channel, channel)
        input_sample_c: np.ndarray = np.reshape(input_sample, (n_in_wo_channel, channel))
        lower_bound_c: np.ndarray = np.reshape(lower_bound, (n_in_wo_channel, channel))
        upper_bound_c: np.ndarray = np.reshape(upper_bound, (n_in_wo_channel, channel))

    gt_label: int = model.predict(input_sample[None], verbose=0).argmax(-1)[0]

    # repeat subdomains
    # freeze xai features to the nominal value
    lower_bound_np: np.ndarray = (
        np.copy(input_sample_c) + 0.0
    )  # (channel, n_in_wo_channel) or (n_in_wo_channel, channel)
    upper_bound_np: np.ndarray = np.copy(input_sample_c) + 0.0
    if len(free_indices):
        if data_format == "channels_first":
            lower_bound_np[:, free_indices] = lower_bound_c[
                :, free_indices
            ]  # open the free dimension
            upper_bound_np[:, free_indices] = upper_bound_c[:, free_indices]
        else:
            lower_bound_np[free_indices, :] = lower_bound_c[
                free_indices, :
            ]  # open the free dimension
            upper_bound_np[free_indices, :] = upper_bound_c[free_indices, :]

    # flatten lower_bound_batch and upper_bound_batch
    lower_bound_batch = np.reshape(
        lower_bound_np, (1, 1, n_in_with_channel)
    )  # (batch_size, 1, n_in_with_channel)
    upper_bound_batch = np.reshape(
        upper_bound_np, (1, 1, n_in_with_channel)
    )  # (batch_size, 1, n_in_with_channel)
    box: np.ndarray = np.concatenate(
        [lower_bound_batch, upper_bound_batch], 1
    )  # (batch_size, 2, n_in_with_channel)

    # build your input domain
    # encode matrix C
    C_gt: np.ndarray = encode_matrix(n_class=n_class, groundtruth=gt_label)[
        None
    ]  # (1, n_class, n_class-1)

    if decomon_model is None:
        C: Tensor = Input((n_class, n_class - 1))

        decomon_model: keras.model.Model = clone(
            model, final_affine=False, final_ibp=True, final_lower=False, backward_bounds=[C]
        )  # return only upper bound

    upper: np.ndarray = decomon_model.predict([box, C_gt], verbose=0)

    return upper
