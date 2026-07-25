from typing import Any, List, Union

import numpy as np
import keras.ops as K
from decomon.perturbation_domain import PerturbationDomain
from keras import KerasTensor as Tensor


def _ensure_weight_batch_axis(
    w: Tensor,
    n_coordinates: int,
    missing_batchsize: bool = False,
) -> Tensor:
    """Normalize affine weights to ``(batch, coordinates, ...)``.

    Decomon may provide an unbatched affine tensor such as
    ``(coordinates, outputs)`` even though the input domain is batched.  Prefer
    Decomon's explicit ``missing_batchsize`` flag, but also infer the condition
    from the coordinate dimension.  The inference protects against wrappers
    that fail to forward the keyword argument.
    """
    rank = len(w.shape)
    first_dim = w.shape[0] if rank else None
    inferred_missing = rank in (1, 2) and first_dim == n_coordinates
    if missing_batchsize or inferred_missing:
        return K.expand_dims(w, axis=0)
    return w


def _feature_energy(
    w: Tensor,
    n_dim: int,
    channel: int,
    data_format: str,
    missing_batchsize: bool = False,
) -> Tensor:
    """Return squared L2 coefficient norm per feature group."""
    w = _ensure_weight_batch_axis(
        w, n_dim * channel, missing_batchsize
    )
    trailing = tuple(w.shape[2:])
    if data_format == "channels_first":
        grouped = K.reshape(w, (-1, channel, n_dim) + trailing)
        return K.sum(K.square(grouped), axis=1)
    if data_format == "channels_last":
        grouped = K.reshape(w, (-1, n_dim, channel) + trailing)
        return K.sum(K.square(grouped), axis=2)
    raise ValueError(f"unknown data format {data_format}")


def _expand_group_mask(mask: Tensor, target_rank: int) -> Tensor:
    return K.reshape(mask, tuple(mask.shape) + (1,) * (target_rank - 2))


def _expand_to_coordinates(mask: Tensor, channel: int, data_format: str) -> Tensor:
    """Expand a (batch, n_features) group mask to flattened coordinates."""
    n_dim = int(mask.shape[1])
    if data_format == "channels_first":
        mask = K.expand_dims(mask, axis=1)
        mask = K.repeat(mask, channel, axis=1)
    elif data_format == "channels_last":
        mask = K.expand_dims(mask, axis=2)
        mask = K.repeat(mask, channel, axis=2)
    else:
        raise ValueError(f"unknown data format {data_format}")
    return K.reshape(mask, (-1, n_dim * channel))


class XAIDomainL2(PerturbationDomain):
    """Hybrid global-L2 / group-L0 domain used by FAME.

    `free_indices` may always vary. Among the remaining non-XAI feature groups,
    at most `cardinalities[b]` groups may vary in batch row b. All varying
    coordinates share one global L2 budget `eps`.

    The [lower, upper] components are used only for sound coordinate-wise IBP
    clipping. Affine concretization uses the L2 support function around center.
    """

    def __init__(
        self,
        xai_indices: List[int],
        free_indices: List[int],
        cardinalities: Union[int, List[int], np.ndarray],
        n_dim: int,
        channel: int,
        eps: float,
        data_format: str = "channels_first",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if data_format not in ("channels_first", "channels_last"):
            raise ValueError(f"unknown data format {data_format}")
        if eps < 0:
            raise ValueError("eps must be non-negative")

        xai = np.zeros((n_dim,), dtype="float32")
        free = np.zeros((n_dim,), dtype="float32")
        xai[np.asarray(xai_indices, dtype=int)] = 1.0
        free[np.asarray(free_indices, dtype=int)] = 1.0
        if np.any((xai + free) > 1.0):
            raise ValueError("xai_indices and free_indices must be disjoint")

        self.n_dim = int(n_dim)
        self.channel = int(channel)
        self.eps = float(eps)
        self.data_format = data_format
        self.xai_mask = xai[None, :]
        self.free_mask = free[None, :]
        self.cardinalities = np.asarray(cardinalities, dtype="int32").reshape(-1)

    def get_nb_x_components(self) -> int:
        return 3

    def get_upper_x(self, x: Tensor) -> Tensor:
        return x[:, 1]

    def get_lower_x(self, x: Tensor) -> Tensor:
        return x[:, 0]

    def get_center_x(self, x: Tensor) -> Tensor:
        return x[:, 2]

    def _support(self, w: Tensor, missing_batchsize: bool = False) -> Tensor:
        energy = _feature_energy(
            w,
            self.n_dim,
            self.channel,
            self.data_format,
            missing_batchsize=missing_batchsize,
        )
        rank = len(energy.shape)
        free_mask = _expand_group_mask(
            K.convert_to_tensor(self.free_mask, dtype=energy.dtype), rank
        )
        xai_mask = _expand_group_mask(
            K.convert_to_tensor(self.xai_mask, dtype=energy.dtype), rank
        )
        candidate_mask = K.clip(1.0 - free_mask - xai_mask, 0.0, 1.0)

        free_energy = K.sum(energy * free_mask, axis=1)
        candidate_energy = energy * candidate_mask
        candidate_energy = K.flip(K.sort(candidate_energy, axis=1), axis=1)

        cards = K.convert_to_tensor(self.cardinalities, dtype="int32")
        card_shape = (-1, 1) + (1,) * (rank - 2)
        cards = K.reshape(cards, card_shape)
        feature_rank = K.reshape(
            K.arange(self.n_dim, dtype="int32"),
            (1, self.n_dim) + (1,) * (rank - 2),
        )
        top_mask = K.cast(feature_rank < cards, energy.dtype)
        candidate_top_energy = K.sum(candidate_energy * top_mask, axis=1)
        total_energy = K.maximum(free_energy + candidate_top_energy, 0.0)
        return K.cast(self.eps, energy.dtype) * K.sqrt(total_energy)

    def _center_value(
        self,
        x_center: Tensor,
        w: Tensor,
        b: Tensor,
        missing_batchsize: bool = False,
    ) -> Tensor:
        w = _ensure_weight_batch_axis(
            w, self.n_dim * self.channel, missing_batchsize
        )
        center = x_center
        while len(center.shape) < len(w.shape):
            center = K.expand_dims(center, axis=-1)
        return K.sum(w * center, axis=1) + b

    def get_upper(self, x: Tensor, w: Tensor, b: Tensor, **kwargs: Any) -> Tensor:
        if len(w.shape) == len(b.shape):
            return self.get_upper_x(x)
        missing_batchsize = bool(kwargs.get("missing_batchsize", False))
        return self._center_value(
            self.get_center_x(x), w, b, missing_batchsize
        ) + self._support(w, missing_batchsize)

    def get_lower(self, x: Tensor, w: Tensor, b: Tensor, **kwargs: Any) -> Tensor:
        if len(w.shape) == len(b.shape):
            return self.get_lower_x(x)
        missing_batchsize = bool(kwargs.get("missing_batchsize", False))
        return self._center_value(
            self.get_center_x(x), w, b, missing_batchsize
        ) - self._support(w, missing_batchsize)


class XAISetDomainL2(PerturbationDomain):
    """Batch of explicit active feature sets under one global L2 budget.

    Row b of `active_masks` specifies exactly which feature groups may vary.
    This is used for efficient, feature-specific singleton refinement.
    """

    def __init__(
        self,
        active_masks: np.ndarray,
        n_dim: int,
        channel: int,
        eps: float,
        data_format: str = "channels_first",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        active_masks = np.asarray(active_masks, dtype="float32")
        if active_masks.ndim != 2 or active_masks.shape[1] != n_dim:
            raise ValueError("active_masks must have shape (batch, n_dim)")
        if np.any((active_masks != 0.0) & (active_masks != 1.0)):
            raise ValueError("active_masks must be binary")
        self.active_masks = active_masks
        self.n_dim = int(n_dim)
        self.channel = int(channel)
        self.eps = float(eps)
        self.data_format = data_format

    def get_nb_x_components(self) -> int:
        return 3

    def get_upper_x(self, x: Tensor) -> Tensor:
        return x[:, 1]

    def get_lower_x(self, x: Tensor) -> Tensor:
        return x[:, 0]

    def get_center_x(self, x: Tensor) -> Tensor:
        return x[:, 2]

    def _group_mask(self, dtype: str) -> Tensor:
        return K.convert_to_tensor(self.active_masks, dtype=dtype)

    def _coordinate_mask(self, dtype: str) -> Tensor:
        return _expand_to_coordinates(
            self._group_mask(dtype), self.channel, self.data_format
        )

    def _center_value(
        self,
        x_center: Tensor,
        w: Tensor,
        b: Tensor,
        missing_batchsize: bool = False,
    ) -> Tensor:
        w = _ensure_weight_batch_axis(
            w, self.n_dim * self.channel, missing_batchsize
        )
        center = x_center
        while len(center.shape) < len(w.shape):
            center = K.expand_dims(center, axis=-1)
        return K.sum(w * center, axis=1) + b

    def _support(self, w: Tensor, missing_batchsize: bool = False) -> Tensor:
        energy = _feature_energy(
            w,
            self.n_dim,
            self.channel,
            self.data_format,
            missing_batchsize=missing_batchsize,
        )
        mask = _expand_group_mask(self._group_mask(energy.dtype), len(energy.shape))
        total_energy = K.maximum(K.sum(energy * mask, axis=1), 0.0)
        return K.cast(self.eps, energy.dtype) * K.sqrt(total_energy)

    def get_upper(self, x: Tensor, w: Tensor, b: Tensor, **kwargs: Any) -> Tensor:
        if len(w.shape) == len(b.shape):
            center = self.get_center_x(x)
            mask = self._coordinate_mask(center.dtype)
            return center + mask * (self.get_upper_x(x) - center)
        missing_batchsize = bool(kwargs.get("missing_batchsize", False))
        return self._center_value(
            self.get_center_x(x), w, b, missing_batchsize
        ) + self._support(w, missing_batchsize)

    def get_lower(self, x: Tensor, w: Tensor, b: Tensor, **kwargs: Any) -> Tensor:
        if len(w.shape) == len(b.shape):
            center = self.get_center_x(x)
            mask = self._coordinate_mask(center.dtype)
            return center + mask * (self.get_lower_x(x) - center)
        missing_batchsize = bool(kwargs.get("missing_batchsize", False))
        return self._center_value(
            self.get_center_x(x), w, b, missing_batchsize
        ) - self._support(w, missing_batchsize)