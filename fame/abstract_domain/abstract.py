from typing import Tuple, Union

import torch
import torch.nn as nn
from typing import Tuple, Optional
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import Perturbation, PerturbationLpNorm

import numpy as np

from ..batch_free.utils import encode_matrix

# Type alias: 'Model' can now be used in any type hint
Model = nn.Module

# Generic TypeVar constrained to nn.Module (useful for generic functions)

Tensor = torch.Tensor




def get_abstract_model(
    model: nn.Module,
    dummy_input: torch.Tensor,
    conv_mode: str = "patches",
    device: Optional[str] = None
) -> BoundedModule:
    """Creates an auto_LiRPA BoundedModule model that computes abstract upper bounds for robustness verification.

    This function takes a standard PyTorch nn.Module and wraps/converts it into an
    auto_LiRPA `BoundedModule`. The resulting model is designed to compute upper and
    lower symbolic bounds for linear combinations of the original model's outputs (logits),
    enabling formal neural network verification and robustness certification via bound
    propagation algorithms (e.g., CROWN, LiRPA, IBP).

    The bound computation leverages an objective specification matrix `C` during backward
    bound propagation. This `C` matrix specifies the linear combinations to verify—typically 
    the pairwise logit differences $z_i - z_j$ for all $j \\neq i$. When evaluated with 
    `compute_bounds(C=C)`, auto_LiRPA computes the lower/upper bounds for these target 
    properties. If the upper bound of $z_j - z_i$ is less than zero (or if the lower bound 
    of $z_i - z_j$ is greater than zero), it proves that class $i$ remains the top prediction.

    Args:
        model: The input PyTorch `nn.Module` to be wrapped into a `BoundedModule`.
        dummy_input: Sample input tensor(s) used by auto_LiRPA to trace the computational graph.
        bound_type: Bound propagation algorithm to prepare or default to (e.g., "CROWN", 
            "IBP", "CROWN-IBP"). Defaults to "CROWN".
        device: Execution device (`torch.device` or str) where the BoundedModule should reside.

    Returns:
        An auto_LiRPA `BoundedModule` instance wrapping the PyTorch model, configured to 
        compute formal output bounds given input perturbation specifications (e.g., `BoundedTensor`) 
        and specification matrices `C`.
    """
    model.eval()

    if device is not None:
        model = model.to(device)
        dummy_input = dummy_input.to(device)
    device = next(model.parameters()).device
    lirpa_model:BoundedModule = BoundedModule(model, torch.empty_like(dummy_input), device=device, bound_opts={'conv_mode': conv_mode})
    lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})

    return lirpa_model
    


def get_abstract_output_domain_singleton(
    model: Model,
    input_shape: Tuple[int],
    gt_label:int, 
    input_sample: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    xai_indices: list[int],
    free_indices: list[int],
    remaining_indices: Union[list[int], None] = None,  # a subset of remaining dimensions or None
    data_format: str = "channels_first",
    n_class: int = 10,
    lirpa_model: BoundedModule = None,
    batch_size: int =2,
    method="CROWN",
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
        lirpa_model: An optional, pre-compiled lirpa model for efficiency.
            If not provided, it will be created internally.

    Returns:
        A numpy array of shape `(len(remaining_indices), n_class - 1)`. Each
        row `i` contains the certified upper bounds on the logit differences
        for the input domain where the `i`-th feature from `remaining_indices`
        was perturbed. A negative value proves robustness for that check.
    """

    if data_format == "channels_first":
        channel, H, W = input_shape
    else:
        H, W, channel = input_shape

    n_in_wo_channel = H * W
    n_in_with_channel = channel * n_in_wo_channel



    # check input sample has batch dimension
    if lirpa_model is None:
        lirpa_model = get_abstract_model(model=model,dummy_input=torch.zeros((1, *input_shape)).to(model.device), device=model.device)

    
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
    perturbation: Perturbation = PerturbationLpNorm(norm=np.inf, 
                                                        eps=np.max(upper_bound - lower_bound))
    for k in range(0, max_batch_size, batch_size):
        # repeat
        k_stop = min(k + batch_size, max_batch_size)
        current_batch_size = k_stop - k
        input_batch: np.ndarray = np.repeat(
            input_sample[None], repeats=current_batch_size, axis=0
        )  # (batch_size, channel, n_in_wo_channel) or (batch_size, n_in_wo_channel, channel)
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
                lower_bound_batch[i, j, :] = lower_bound_c[j, :]
                upper_bound_batch[i, j, :] = upper_bound_c[j, :]


        # flatten lower_bound_batch and upper_bound_batch
        lower_bound_batch = np.reshape(lower_bound_batch, (-1, *input_shape))  # (batch_size, *input_shape)
        upper_bound_batch = np.reshape(upper_bound_batch, (-1, *input_shape))
        input_batch = np.reshape(input_batch, (-1, *input_shape))

        # build your input domain
        # encode matrix C
        C_gt: np.ndarray = np.repeat(
            encode_matrix(n_class=n_class, groundtruth=gt_label)[None], repeats=current_batch_size, axis=0
        )
        # (batch_size, n_class, n_class-1)
        C_gt = np.transpose(C_gt, (0, 2, 1))  # (current_batch_size, n_class-1, n_class)
        

        device = next(model.parameters()).device
        # convert into torch tensor
        upper_bound_batch: Tensor = torch.from_numpy(upper_bound_batch).float().to(
            device
        )
        lower_bound_batch: Tensor = torch.from_numpy(lower_bound_batch).float().to(
            device
        )   
        input_batch: Tensor = torch.from_numpy(input_batch).float().to(device)

        # compute lirpa with Linf Norm Perturbation
        bounded_image = BoundedTensor(input_batch, perturbation)
        perturbation.x_L = lower_bound_batch
        perturbation.x_U = upper_bound_batch
        lb_output, ub_output = lirpa_model.compute_bounds(x=(bounded_image,), method=method, C=torch.from_numpy(C_gt).float().to(device), bound_lower=False, bound_upper=True)

        all_upper.append(ub_output.detach().cpu().numpy())

    upper = np.concatenate(all_upper, axis=0)
    return upper



def get_abstract_output_domain(
    model: Model,
    input_sample: Tensor,
    C: Tensor,
    perturbation: Perturbation, 
    affine_bounds:bool=True,
    data_format: str = "channels_first",
    method="CROWN",
    lirpa_model: BoundedModule = None,
    bound_upper:bool=True,
    bound_lower:bool=False,
) -> tuple[np.ndarray]:
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


    if lirpa_model is None:
        lirpa_model = get_abstract_model(model=model,dummy_input=input_sample[:1]).to(model.device)

    bounded_image = BoundedTensor(input_sample, perturbation)

    if affine_bounds:
        input_node_name = lirpa_model.input_name[0]  # Or list key from lirpa_model.nodes_dict

        # 2. Specify that we need the A matrices for output node with respect to input node
        needed_A_dict = {lirpa_model.output_name[0]: [input_node_name]}  
        lower_output, upper_output, A_dict = lirpa_model.compute_bounds(x=(bounded_image,),method=method, \
                                                        return_A=True, needed_A_dict=needed_A_dict, C=C,\
                                                        bound_lower=bound_lower, bound_upper=bound_upper)
        A_info = A_dict[lirpa_model.output_name[0]][input_node_name]

        if bound_upper:
            A = A_info['uA']     # Upper affine slope matrix,
            b = A_info['ubias']  # Upper affine bias,
        else:
            A = A_info['lA']     # Lower affine slope matrix,
            b = A_info['lbias']  # Lower affine bias,
        
        # detach and cast to numpy arrays

        w = A.detach().cpu().numpy()
        b = b.detach().cpu().numpy()    

        # reshape w_u to (batch, n_in_with_channel, n_class-1)
        w = np.reshape(w, (w.shape[0], -1, b.shape[1]))

        if bound_upper:
            return w, b, upper_output.detach().cpu().numpy()
        else:
            return w, b, lower_output.detach().cpu().numpy()
    else:
        if bound_upper:
            _, ub_output = lirpa_model.compute_bounds(x=(bounded_image,),method=method, C=C, bound_lower=False, bound_upper=True)
            return ub_output.detach().cpu().numpy()
        else:
            lb_output, _ = lirpa_model.compute_bounds(x=(bounded_image,),method=method, C=C, bound_lower=True, bound_upper=False)
            return lb_output.detach().cpu().numpy()



