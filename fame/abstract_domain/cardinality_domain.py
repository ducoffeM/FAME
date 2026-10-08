from typing import Any, List, Union


import numpy as np
import math
import torch
from auto_LiRPA.perturbations import PerturbationLpNorm as Perturbation
from auto_LiRPA.patches import Patches
#from auto_LiRPA.patches import inplace_unfold
import copy
import torch


class XAIDomain(Perturbation):
    """Defines a hybrid spatial abstract domain for features perturbed under combined L-infinity and spatial L0 norms.

    This class implements a custom `auto_LiRPA` `Perturbation` domain specifically for formal eXplainable AI (XAI)
    verification and abductive explanation framework (FAME). It models a complex input space where:
    1.  A set of "free" spatial locations (`free_mask`) can always vary within their explicit physical lower and
        upper bounds [x_L, x_U] under an L-infinity perturbation model (costing 0 units of the L0 budget).
    2.  A set of candidate spatial locations (`xai_mask`) are eligible for perturbation, but at most `eps` 
        spatial locations (i, j) across all channels can be perturbed simultaneously (spatial L0-norm constraint).
    3.  All other features (where both masks are 0) remain fixed to their nominal input values `x`.

    The domain seamlessly handles both 2D dense linear coefficient matrices and 4D sparse `Patches` representations,
    aggregating worst-case channel deviations locally in patch space to prevent GPU memory allocation explosions (OOM).

    Attributes:
        eps (int, float, list, or torch.Tensor): The L0-norm spatial budget, i.e., the maximum number of spatial 
            locations (i, j) allowed to be perturbed per sample. Can be a scalar or a 1D tensor of shape (batch_size,).
        xai_mask (torch.Tensor, optional): A binary mask indicating which spatial/feature locations are candidates 
            for L0-norm perturbation. Shared across batch if shaped (1, C, H, W) or (C, H, W).
        free_mask (torch.Tensor, optional): A binary mask indicating which spatial/feature locations are always 
            perturbed under L-infinity bounds without consuming L0 budget. Shared across batch if shaped (1, C, H, W).
        x_L (torch.Tensor, optional): The explicit physical lower bounds for input features. Defaults to torch.zeros_like(x).
        x_U (torch.Tensor, optional): The explicit physical upper bounds for input features. Defaults to torch.ones_like(x).
        ratio (float): A bound scaling parameter used for scheduling/relaxing bounds during certified training.
        data_format (str): The spatial layout of 4D tensors, accepting "channels_first" ("nchw") or "channels_last" ("nhwc").
    """

    def __init__(self, eps, x_L, x_U,xai_mask=None, free_mask=None, data_format="channels_first"):

        """
        # 1. Standardize eps into a 1D LongTensor of shape (B,)
        if isinstance(eps, torch.Tensor):
            eps_tensor = torch.ceil(eps).long()
        elif isinstance(eps, (list, tuple)):
            eps_tensor = torch.tensor([math.ceil(e) for e in eps], dtype=torch.long)
        else:
            eps_tensor = torch.tensor([math.ceil(eps)], dtype=torch.long)

        # 2. Count distinct spatial locations in free_mask
        num_free_spatial = 0
        if free_mask is not None:
            num_free_spatial = int(free_mask.sum().item())  # Count total free locations
            
        # 3. Compute expanded budget scalar for super() initializer
        max_eps_scalar = int(eps_tensor.max().item()) + num_free_spatial
        """

        super().__init__(eps=torch.max(x_U-x_L).view(-1).max().item(), norm=float('inf'), x_L=x_L, x_U=x_U)
        if isinstance(eps, torch.Tensor):  # eps shape: (B,)
            self.eps_ = torch.ceil(eps).long()  # self.eps shape: (B,)
        elif isinstance(eps, (list, tuple)):  # len(eps) = B
            self.eps_ = torch.tensor([math.ceil(e) for e in eps], dtype=torch.long)  # self.eps shape: (B,)
        else:  # Scalar int/float
            self.eps_ = eps  # self.eps shape: scalar

        self.xai_mask = xai_mask  # self.xai_mask shape: (H, W) or None
        self.free_mask = free_mask  # self.free_mask shape: (H, W) or None
        self.data_format = data_format.lower()  # string

    def set_xai_mask(self, xai_mask):
        """Sets the xai_mask attribute."""
        self.xai_mask = xai_mask  # self.xai_mask shape: (H, W)

    def set_free_mask(self, free_mask): 
        """Sets the free_mask attribute."""
        self.free_mask = free_mask  # self.free_mask shape: (H, W)

    def set_bounds(self, x_L, x_U): 
        """Sets the explicit physical bounds for input features."""
        self.x_L = x_L  # self.x_L shape: (B, C, H, W)
        self.x_U = x_U  # self.x_U shape: (B, C, H, W)


    def concretize_patches_card(self, A, x, x_L, x_U, xai_full, free_full, sign=1):
        """
        Line-by-line symbolic shape tracking.
        """

        batch_size = x.shape[0]                                             # batch_size = B

        # 1. Compute Center Output
        # Matrix mapping: A: (B, C, H, W) -> (B, out_dim)
        center = A.matmul(x)                                                # center shape: (B, unstable_size)

        # 2. Compute Input Perturbation Deltas
        # Element-wise subtraction between tensors of shape (B, C, H, W)
        diff_upper = x_U - x   #>=0                                             # diff_upper shape: (B, C, H, W)
        diff_lower = x - x_L    #>=0                                            # diff_lower shape: (B, C, H, W)


        # 3. Apply Domain Masks directly in Image Space
        # Broadcast/element-wise multiplication: (B, C, H, W) * (B, C, H, W) -> (B, C, H, W)

        if sign == 1:  # Upper Bound Mode
            free_upper_delta = diff_upper * free_full                       # shape: (B, C, H, W)
            free_lower_delta = diff_lower * free_full                       # shape: (B, C, H, W)
            
            cand_upper_delta = diff_upper * (1.0-xai_full) * (1.0 - free_full)    # shape: (B, C, H, W)
            cand_lower_delta = diff_lower * (1.0-xai_full) * (1.0 - free_full)    # shape: (B, C, H, W)

            

        else:  # Lower Bound Mode
            free_upper_delta = diff_lower * free_full                       # shape: (B, C, H, W)
            free_lower_delta = diff_upper * free_full                       # shape: (B, C, H, W)
            
            cand_upper_delta = diff_lower * (1.0-xai_full) * (1.0 - free_full)    # shape: (B, C, H, W)
            cand_lower_delta = diff_upper * (1.0-xai_full) * (1.0 - free_full)    # shape: (B, C, H, W)

        # 4. Native auto_LiRPA Matrix Propagation
        # Linear mapping maps image domain (B, C, H, W) to feature/output domain (B, out_dim, H, W) or (B, out_dim)
        pos_A, neg_A = split_patches(A)  # pos_A and neg_A are Patches objects with positive and negative weights

        free_spatial_grid = pos_A.matmul(free_upper_delta) - neg_A.matmul(free_lower_delta)   #>=0                          # shape: (B, unstable_size)
        candidate_spatial_grid = pos_A.matmul(cand_upper_delta) - neg_A.matmul(cand_lower_delta)     #>=0                       # shape: (B, unstable_size)


        
        # 5. Format Spatial Features
        # Reshaping spatial dimensions (H, W) -> N_in = (H * W)
        free_spatial = free_spatial_grid.unsqueeze(-1)                                 # shape: (B, unstable_size, 1)
        candidate_spatial = candidate_spatial_grid.unsqueeze(-1)                       # shape: (B, unstable_size, 1)

        # 6. Aggregate Total Free Bound & Flatten Candidate Map
        # Sum over the flattened spatial dimension dim=-1: (B, unstable_size, H * W) -> (B, unstable_size, 1)
        free_total = free_spatial.sum(dim=-1, keepdim=True)                                # shape: (B, unstable_size, 1)

        center = center.unsqueeze(-1)                                                      # shape: (B, unstable_size, 1)

        unstable_size = np.prod(center.shape)//batch_size  # out_dim = unstable_size

        candidate_spatial = candidate_spatial.reshape(batch_size, unstable_size, -1)  # shape: (B, unstable_size, H * W)
        free_spatial = free_spatial.reshape(batch_size, unstable_size, -1)              # shape: (B, unstable_size, H * W)
        center = center.reshape(batch_size, unstable_size, -1)                              # shape: (B, unstable_size, 1)


        
        return candidate_spatial, free_spatial, center                                    # tuple: ((B, unstable_size, H * W), (B, unstable_size, 1), (B, unstable_size, 1))



    def concretize_dense_card(self, A, x, x_L, x_U, xai_full, free_full, sign=1):

        batch_size = x.shape[0]  # B (scalar int)
        out_dim = np.prod(A.shape)//(np.prod(x.shape))  # out_dim (scalar int)

        num_features = A.shape[2]  # num_features = C * H * W (scalar int)

        x_flat = x.reshape(batch_size, -1)  # x_flat shape: (B, num_features)
        x_L_flat = x_L.reshape(batch_size, -1)  # x_L_flat shape: (B, num_features)
        x_U_flat = x_U.reshape(batch_size, -1)  # x_U_flat shape: (B, num_features)

        xai_flat = xai_full.reshape(batch_size, -1).unsqueeze(1)  # xai_flat shape: (B, 1, num_features)
        free_flat = free_full.reshape(batch_size, -1).unsqueeze(1)  # free_flat shape: (B, 1, num_features)

        center = A.matmul(x_flat.unsqueeze(-1))  # center shape: (B, out_dim, 1)

        x_expand = x_flat.unsqueeze(1).expand(batch_size, out_dim, num_features)  # x_expand shape: (B, out_dim, num_features)
        x_L_expand = x_L_flat.unsqueeze(1).expand(batch_size, out_dim, num_features)  # x_L_expand shape: (B, out_dim, num_features)
        x_U_expand = x_U_flat.unsqueeze(1).expand(batch_size, out_dim, num_features)  # x_U_expand shape: (B, out_dim, num_features)

        pos_mask = (A >= 0)  # pos_mask shape: (B, out_dim, num_features) [bool]
        neg_mask = (A < 0)  # neg_mask shape: (B, out_dim, num_features) [bool]

        A_diff = torch.zeros_like(A)  # A_diff shape: (B, out_dim, num_features)
        if sign == 1:  # Upper bound mode
            A_diff[pos_mask] = A[pos_mask] * (x_U_expand - x_expand)[pos_mask]  # Slicing assignment
            A_diff[neg_mask] = A[neg_mask] * (x_L_expand - x_expand)[neg_mask]  # Slicing assignment
        else:  # Lower bound mode
            A_diff[pos_mask] = A[pos_mask] * (x_expand - x_L_expand)[pos_mask]  # Slicing assignment
            A_diff[neg_mask] = A[neg_mask] * (x_expand - x_U_expand)[neg_mask]  # Slicing assignment

        free_diff_raw = A_diff * free_flat  # free_diff_raw shape: (B, out_dim, num_features)
        # l0 candidate are neitgher in xai_indices not in free_indices, so we multiply by (1 - xai_flat) * (1 - free_flat)
        l0_candidate_diff_raw = A_diff * (1- xai_flat) * (1.0 - free_flat)  # l0_candidate_diff_raw shape: (B, out_dim, num_features)

        if x.dim() == 4:  # 4D Image check
            if self.data_format in ["channels_first", "nchw"]:  # NCHW layout
                c, h, w = x.shape[1], x.shape[2], x.shape[3]  # Scalars: C, H, W
                free_reshaped = free_diff_raw.reshape(batch_size, out_dim, c, h * w)  # free_reshaped shape: (B, out_dim, C, H*W)
                candidate_reshaped = l0_candidate_diff_raw.reshape(batch_size, out_dim, c, h * w)  # candidate_reshaped shape: (B, out_dim, C, H*W)
                free_spatial = free_reshaped.sum(dim=2)  # free_spatial shape: (B, out_dim, H*W)
                candidate_spatial = candidate_reshaped.sum(dim=2)  # candidate_spatial shape: (B, out_dim, H*W)
            else:  # NHWC layout
                h, w, c = x.shape[1], x.shape[2], x.shape[3]  # Scalars: H, W, C
                free_reshaped = free_diff_raw.reshape(batch_size, out_dim, h * w, c)  # free_reshaped shape: (B, out_dim, H*W, C)
                candidate_reshaped = l0_candidate_diff_raw.reshape(batch_size, out_dim, h * w, c)  # candidate_reshaped shape: (B, out_dim, H*W, C)
                free_spatial = free_reshaped.sum(dim=-1)  # free_spatial shape: (B, out_dim, H*W)
                candidate_spatial = candidate_reshaped.sum(dim=-1)  # candidate_spatial shape: (B, out_dim, H*W)
        else:  # Fallback 1D
            free_spatial = free_diff_raw  # free_spatial shape: (B, out_dim, num_features)
            candidate_spatial = l0_candidate_diff_raw  # candidate_spatial shape: (B, out_dim, num_features)

        free_total = free_spatial.sum(dim=2).unsqueeze(2)  # free_total shape: (B, out_dim, 1)
        candidate_flat = candidate_spatial  # candidate_flat shape: (B, out_dim, H*W)

        return candidate_flat, free_total, center # (B, out_dim, H*W), (B, out_dim, 1), (B, out_dim, 1)

    def prepare_mask(self, x, mask, device=None):
        """
        Expands a 2D spatial mask (H, W) to match full 4D input x shape:
        - (B, C, H, W) for 'channels_first' / 'nchw'
        - (B, H, W, C) for 'channels_last' / 'nhwc'
        """
        batch_size = x.shape[0]

        if mask is None:
            return torch.zeros_like(x)

        # Detach and strip trailing/leading singleton dimensions (e.g. if passed 1xHxW or HxWx1)
        m = mask.detach().squeeze()  # Shape: (H, W)

        # 1. Expand spatial mask across channels C based on data_format
        if self.data_format in ["channels_first", "nchw"]:
            # (H, W) -> (1, H, W) -> (C, H, W)
            m_sample = m.unsqueeze(0).expand(x.shape[1], *m.shape)
        else:
            # (H, W) -> (H, W, 1) -> (H, W, C)
            m_sample = m.unsqueeze(-1).expand(*m.shape, x.shape[-1])

        # 2. Unsqueeze batch dimension and force expansion to full batch size B
        # (C, H, W) -> (1, C, H, W) -> (B, C, H, W)
        if device is not None:
            return m_sample.unsqueeze(0).expand(batch_size, *m_sample.shape).to(device)
        return m_sample.unsqueeze(0).expand(batch_size, *m_sample.shape)


    def concretize(self, x, A, sign=-1, aux=None):

        bound_ = super().concretize(x, A, sign=sign, aux=aux)
        return bound_

        batch_size = x.shape[0]  # B (scalar int)
        x_L = self.x_L if self.x_L is not None else torch.zeros_like(x)  # x_L shape: (B, C, H, W)
        x_U = self.x_U if self.x_U is not None else torch.ones_like(x)  # x_U shape: (B, C, H, W)
        xai_full = self.prepare_mask(x, self.xai_mask, device=x.device)  # xai_full shape: (B, C, H, W)
        free_full = self.prepare_mask(x, self.free_mask, device=x.device)  # free_full shape: (B, C, H, W)

        # =========================================================================
        # PATH 1: Native Patches Mode (Memory Efficient - Prevents OOM)
        # =========================================================================
        if isinstance(A, Patches):  # Patches check

            # raise error if data_format is not channel_first
            if self.data_format not in ["channels_first", "nchw"]:
                raise ValueError("Patches concretization requires 'channels_first' / 'nchw' data format.")
            candidate_flat, free_total, center = self.concretize_patches_card(A, x, x_L, x_U, xai_full, free_full, sign=sign)  
            # candidate_flat shape: (B, out_dim, S), free_total shape: (B, out_dim, 1)
        # =========================================================================
        # PATH 2: Dense Matrix Fallback
        # =========================================================================
        else:
            candidate_flat, free_total, center = self.concretize_dense_card(A, x, x_L, x_U, xai_full, free_full, sign=sign) 
            # candidate_flat shape: (B, out_dim, S), free_total shape: (B, out_dim, 1)

        # =========================================================================
        # Shared Top-K Selection across Spatial Locations
        # =========================================================================
        candidate_sorted, _ = torch.sort(candidate_flat, dim=-1, descending=True)  # candidate_sorted shape: (B, out_dim, S)

        if isinstance(self.eps_, torch.Tensor):  # Batch-wise eps
            self.eps = self.eps_.to(A.device)  # self.eps shape: (B,)
            max_k = candidate_sorted.shape[-1]  # Scalar: S = H * W
            idx = torch.arange(max_k, device=A.device).unsqueeze(0)  # idx shape: (1, S)
            top_k_mask = (idx < self.eps_.unsqueeze(1)).unsqueeze(1)  # top_k_mask shape: (B, 1, S)

            l0_total = (candidate_sorted * top_k_mask).sum(dim=-1, keepdim=True)  # l0_total shape: (B, out_dim, 1)
        else:  # Scalar eps
            eps_val = math.ceil(self.eps_)  # Scalar int
            l0_total = candidate_sorted[:, :, :eps_val].sum(dim=-1, keepdim=True)  # l0_total shape: (B, out_dim, 1)

        total_diff = free_total + l0_total  # total_diff shape: (B, out_dim, 1)
        bound = (center + sign * total_diff).squeeze(-1)  # bound shape: (B, out_dim)

        bound.view(bound.shape[0], -1)  # Return shape: (B, out_dim)

        return bound


def split_patches(A):
    """Splits a Patches object A into positive (pos_A) and negative (neg_A) Patches objects."""
    # 1. Create shallow copies to preserve Patches methods and metadata
    pos_A = copy.copy(A)
    neg_A = copy.copy(A)

    # 2. Extract the raw tensor weights from A.patches
    w = A.patches

    # 3. Apply abs/decompositions on the tensor itself
    pos_A.patches = torch.relu(w)
    neg_A.patches = -torch.relu(-w)

    return pos_A, neg_A

