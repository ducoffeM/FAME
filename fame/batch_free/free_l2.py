from __future__ import annotations

from typing import Optional, Sequence, Tuple

import keras
import numpy as np
from decomon import clone
from keras import KerasTensor as Tensor
from keras.layers import Input

from fame.abstract_domain.cardinality_domain_l2 import XAIDomainL2, XAISetDomainL2
from fame.batch_free.utils import encode_matrix


def _validate_layout(input_sample: np.ndarray, channel: int) -> int:
    if input_sample.ndim != 1:
        raise ValueError("FAME expects a flattened input_sample")
    if channel <= 0 or input_sample.shape[-1] % channel != 0:
        raise ValueError("input dimension must be divisible by channel")
    return input_sample.shape[-1] // channel


def _make_box(
    input_sample: np.ndarray,
    eps: float,
    means: Optional[np.ndarray] = None,
    stddev: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Coordinate box used only as a sound IBP bound for the clipped L2 ball."""
    x = np.asarray(input_sample, dtype="float32")
    if means is None or stddev is None:
        return np.maximum(x - eps, 0.0), np.minimum(x + eps, 1.0)
    return (
        np.maximum(x - eps, -(means / stddev)),
        np.minimum(x + eps, (1.0 - means) / stddev),
    )


def _freeze_groups_in_box(
    lower: np.ndarray,
    upper: np.ndarray,
    center: np.ndarray,
    fixed_indices: Sequence[int],
    channel: int,
    data_format: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not fixed_indices:
        return lower, upper
    n_dim = center.shape[-1] // channel
    fixed = np.asarray(sorted(set(fixed_indices)), dtype=int)
    if data_format == "channels_first":
        lo = lower.reshape(channel, n_dim)
        hi = upper.reshape(channel, n_dim)
        x = center.reshape(channel, n_dim)
        lo[:, fixed] = x[:, fixed]
        hi[:, fixed] = x[:, fixed]
    elif data_format == "channels_last":
        lo = lower.reshape(n_dim, channel)
        hi = upper.reshape(n_dim, channel)
        x = center.reshape(n_dim, channel)
        lo[fixed, :] = x[fixed, :]
        hi[fixed, :] = x[fixed, :]
    else:
        raise ValueError(f"unknown data format {data_format}")
    return lo.reshape(-1), hi.reshape(-1)


def _predict_with_domain(
    model: keras.models.Model,
    gt_label: int,
    box: np.ndarray,
    domain,
    n_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch = box.shape[0]
    c_input: Tensor = Input((n_class, n_class - 1))
    c_gt = np.repeat(
        encode_matrix(n_class=n_class, groundtruth=gt_label)[None],
        repeats=batch,
        axis=0,
    )
    decomon_model = clone(
        model,
        perturbation_domain=domain,
        final_affine=True,
        final_ibp=True,
        final_lower=False,
        backward_bounds=[c_input],
    )
    w_u, b_u, upper = decomon_model.predict(
        [box, c_gt], verbose=0, batch_size=batch
    )
    return np.asarray(w_u), np.asarray(b_u), np.asarray(upper)


def get_features_batch_l2(
    model: keras.models.Model,
    gt_label: int,
    input_sample: np.ndarray,
    lower_bound_input: np.ndarray,
    upper_bound_input: np.ndarray,
    xai_indices: Sequence[int],
    free_indices: Sequence[int],
    cardinality: np.ndarray,
    eps: float,
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    batch_size: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CROWN bounds for free features + at most k additional feature groups."""
    n_dim = _validate_layout(input_sample, channel)
    cardinality = np.asarray(cardinality, dtype="int32").reshape(-1)
    if cardinality.size == 0:
        raise ValueError("cardinality must be non-empty")
    batch_size = min(max(int(batch_size), 1), cardinality.size)

    all_w, all_b, all_upper, all_box = [], [], [], []
    for start in range(0, cardinality.size, batch_size):
        stop = min(start + batch_size, cardinality.size)
        cards = cardinality[start:stop]
        current = cards.size
        lower = np.repeat(lower_bound_input[None], current, axis=0)
        upper = np.repeat(upper_bound_input[None], current, axis=0)
        center = np.repeat(input_sample[None], current, axis=0)
        box = np.stack([lower, upper, center], axis=1)
        domain = XAIDomainL2(
            xai_indices=list(xai_indices),
            free_indices=list(free_indices),
            cardinalities=cards,
            n_dim=n_dim,
            channel=channel,
            eps=eps,
            data_format=data_format,
        )
        w_u, b_u, upper_out = _predict_with_domain(
            model, gt_label, box, domain, n_class
        )
        all_w.append(w_u)
        all_b.append(b_u)
        all_upper.append(upper_out)
        all_box.append(box)

    return (
        np.concatenate(all_w, axis=0),
        np.concatenate(all_b, axis=0),
        np.concatenate(all_upper, axis=0),
        np.concatenate(all_box, axis=0),
    )


def get_explicit_sets_batch_l2(
    model: keras.models.Model,
    gt_label: int,
    input_sample: np.ndarray,
    lower_bound_input: np.ndarray,
    upper_bound_input: np.ndarray,
    active_masks: np.ndarray,
    eps: float,
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CROWN bounds for a batch of explicit allowed feature sets."""
    n_dim = _validate_layout(input_sample, channel)
    active_masks = np.asarray(active_masks, dtype="float32")
    if active_masks.ndim != 2 or active_masks.shape[1] != n_dim:
        raise ValueError("active_masks must have shape (batch, n_features)")

    all_w, all_b, all_upper = [], [], []
    for start in range(0, active_masks.shape[0], batch_size):
        stop = min(start + batch_size, active_masks.shape[0])
        masks = active_masks[start:stop]
        current = masks.shape[0]
        lower = np.repeat(lower_bound_input[None], current, axis=0)
        upper = np.repeat(upper_bound_input[None], current, axis=0)
        center = np.repeat(input_sample[None], current, axis=0)
        box = np.stack([lower, upper, center], axis=1)
        domain = XAISetDomainL2(
            active_masks=masks,
            n_dim=n_dim,
            channel=channel,
            eps=eps,
            data_format=data_format,
        )
        w_u, b_u, upper_out = _predict_with_domain(
            model, gt_label, box, domain, n_class
        )
        all_w.append(w_u)
        all_b.append(b_u)
        all_upper.append(upper_out)
    return (
        np.concatenate(all_w, axis=0),
        np.concatenate(all_b, axis=0),
        np.concatenate(all_upper, axis=0),
    )


def _feature_energy_numpy(
    w_u: np.ndarray,
    channel: int,
    data_format: str,
) -> np.ndarray:
    """Squared coefficient L2 norm per feature group.

    Input: (batch, coordinates, outputs)
    Output: (batch, feature_groups, outputs)
    """
    batch, n_coords = w_u.shape[:2]
    n_dim = n_coords // channel
    trailing = w_u.shape[2:]
    if data_format == "channels_first":
        grouped = w_u.reshape((batch, channel, n_dim) + trailing)
        return np.sum(np.square(grouped, dtype=np.float64), axis=1)
    if data_format == "channels_last":
        grouped = w_u.reshape((batch, n_dim, channel) + trailing)
        return np.sum(np.square(grouped, dtype=np.float64), axis=2)
    raise ValueError(f"unknown data format {data_format}")


def _greedy_l2_from_affine(
    w_u: np.ndarray,
    b_u: np.ndarray,
    upper: np.ndarray,
    input_sample: np.ndarray,
    cardinality: np.ndarray,
    eps: float,
    xai_indices: Sequence[int],
    free_indices: Sequence[int],
    channel: int,
    data_format: str,
    certificate_margin: float,
) -> np.ndarray:
    """Greedy MKP for the squared-energy L2 certificate.

    Critical safety rule: a row is never squared unless every center baseline is
    strictly negative. This prevents the classic all-features bug.
    """
    w = np.asarray(w_u, dtype=np.float64)
    b = np.asarray(b_u, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    x = np.asarray(input_sample, dtype=np.float64)
    cards = np.asarray(cardinality, dtype=int).reshape(-1)
    n_dim = x.size // channel
    result = np.zeros((cards.size, n_dim), dtype="float32")

    energies = _feature_energy_numpy(w, channel, data_format)
    baseline = np.sum(w * x[None, :, None], axis=1) + b
    xai = set(int(i) for i in xai_indices)
    free = set(int(i) for i in free_indices)
    candidates = np.array(
        [j for j in range(n_dim) if j not in xai and j not in free], dtype=int
    )
    if candidates.size == 0:
        return result

    for row, k_requested in enumerate(cards):
        k = min(max(int(k_requested), 0), candidates.size)
        if k == 0:
            continue

        # If the whole sparse domain is already certified, every subset of size
        # <= k is safe. Use the affine energy only to choose a stable subset.
        if np.max(upper[row]) <= -certificate_margin:
            score = np.max(energies[row, candidates, :], axis=1)
            chosen = candidates[np.argsort(score, kind="stable")[:k]]
            result[row, chosen] = 1.0
            continue

        center_margin = baseline[row]
        # DO NOT square a nonnegative baseline. That creates a false capacity.
        if np.any(center_margin >= -certificate_margin):
            continue

        free_energy = (
            np.sum(energies[row, list(free), :], axis=0)
            if free
            else np.zeros_like(center_margin)
        )

        if eps == 0.0:
            chosen = candidates[:k]
            result[row, chosen] = 1.0
            continue

        radius_to_boundary = (-center_margin - certificate_margin) / eps
        capacity = np.square(radius_to_boundary) - free_energy
        if np.any(capacity < -1e-12):
            continue
        remaining_capacity = np.maximum(capacity, 0.0)

        remaining = candidates.copy()
        selected: list[int] = []
        selected_energy = np.zeros_like(center_margin)

        for _ in range(k):
            if remaining.size == 0:
                break
            costs = energies[row, remaining, :]
            feasible = np.all(costs <= remaining_capacity[None, :] + 1e-12, axis=1)
            if not np.any(feasible):
                break
            feasible_pos = np.flatnonzero(feasible)
            feasible_costs = costs[feasible]
            denom = np.maximum(remaining_capacity, 1e-30)
            scores = np.max(feasible_costs / denom[None, :], axis=1)
            local = feasible_pos[int(np.argmin(scores))]
            feature = int(remaining[local])
            cost = energies[row, feature, :]

            # Independent certificate check before committing the feature.
            trial_energy = selected_energy + cost
            trial_upper = center_margin + eps * np.sqrt(
                np.maximum(free_energy + trial_energy, 0.0)
            )
            if np.max(trial_upper) > -certificate_margin + 1e-10:
                remaining = np.delete(remaining, local)
                continue

            selected.append(feature)
            selected_energy = trial_energy
            remaining_capacity = np.maximum(remaining_capacity - cost, 0.0)
            remaining = np.delete(remaining, local)

        if selected:
            final_upper = center_margin + eps * np.sqrt(
                np.maximum(free_energy + selected_energy, 0.0)
            )
            if np.max(final_upper) > -certificate_margin + 1e-10:
                raise RuntimeError("internal L2 certificate check failed")
            result[row, selected] = 1.0

    return result


def free_at_once_k_features_l2(
    model: keras.models.Model,
    gt_label: int,
    input_sample: np.ndarray,
    lower_bound_input: np.ndarray,
    upper_bound_input: np.ndarray,
    eps: float,
    xai_indices: Optional[Sequence[int]] = None,
    free_indices: Optional[Sequence[int]] = None,
    cardinality: Optional[np.ndarray] = None,
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    verbose: int = 0,
    certificate_margin: float = 1e-6,
) -> np.ndarray:
    xai_indices = list(xai_indices or [])
    free_indices = list(free_indices or [])
    n_dim = _validate_layout(input_sample, channel)
    if cardinality is None:
        cardinality = np.arange(1, n_dim - len(free_indices) + 1, dtype=int)
    cardinality = np.asarray(cardinality, dtype=int).reshape(-1)

    lower = np.asarray(lower_bound_input, dtype="float32").copy()
    upper = np.asarray(upper_bound_input, dtype="float32").copy()
    lower, upper = _freeze_groups_in_box(
        lower, upper, np.asarray(input_sample), xai_indices, channel, data_format
    )
    w_u, b_u, upper_out, _ = get_features_batch_l2(
        model=model,
        gt_label=gt_label,
        input_sample=np.asarray(input_sample, dtype="float32"),
        lower_bound_input=lower,
        upper_bound_input=upper,
        xai_indices=xai_indices,
        free_indices=free_indices,
        cardinality=cardinality,
        eps=eps,
        channel=channel,
        data_format=data_format,
        n_class=n_class,
    )
    if verbose:
        print("L2 upper:", upper_out)
    return _greedy_l2_from_affine(
        w_u=w_u,
        b_u=b_u,
        upper=upper_out,
        input_sample=input_sample,
        cardinality=cardinality,
        eps=eps,
        xai_indices=xai_indices,
        free_indices=free_indices,
        channel=channel,
        data_format=data_format,
        certificate_margin=certificate_margin,
    )


def free_with_singleton_search_l2(
    model: keras.models.Model,
    gt_label: int,
    input_sample: np.ndarray,
    lower_bound_input: np.ndarray,
    upper_bound_input: np.ndarray,
    eps: float,
    free_indices: Sequence[int],
    xai_indices: Optional[Sequence[int]] = None,
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    certificate_margin: float = 1e-6,
    batch_size: int = 64,
) -> list[int]:
    """Feature-specific singleton refinement; adds only one feature per pass."""
    n_dim = _validate_layout(input_sample, channel)
    fixed = set(int(i) for i in (xai_indices or []))
    current_free = set(int(i) for i in free_indices)
    added: list[int] = []

    while True:
        candidates = [
            j for j in range(n_dim) if j not in fixed and j not in current_free
        ]
        if not candidates:
            break
        active = np.zeros((len(candidates), n_dim), dtype="float32")
        if current_free:
            active[:, list(current_free)] = 1.0
        active[np.arange(len(candidates)), candidates] = 1.0

        _, _, upper = get_explicit_sets_batch_l2(
            model=model,
            gt_label=gt_label,
            input_sample=input_sample,
            lower_bound_input=lower_bound_input,
            upper_bound_input=upper_bound_input,
            active_masks=active,
            eps=eps,
            channel=channel,
            data_format=data_format,
            n_class=n_class,
            batch_size=batch_size,
        )
        worst = np.max(np.asarray(upper, dtype=np.float64), axis=1)
        safe = np.flatnonzero(worst <= -certificate_margin)
        if safe.size == 0:
            break

        # One at a time: freeing all individually safe features together is unsound.
        best_pos = int(safe[np.argmin(worst[safe])])
        best_feature = int(candidates[best_pos])
        current_free.add(best_feature)
        added.append(best_feature)

    return added


def free_iteratively_k_features_l2(
    model: keras.models.Model,
    gt_label: int,
    input_sample: np.ndarray,
    eps: float,
    xai_indices: Optional[Sequence[int]] = None,
    free_indices: Optional[Sequence[int]] = None,
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    refining_domain: bool = True,
    singleton_refinement: bool = True,
    verbose: int = 0,
    means: Optional[np.ndarray] = None,
    stddev: Optional[np.ndarray] = None,
    certificate_margin: float = 1e-6,
) -> tuple[list[int], list[int]]:
    n_dim = _validate_layout(input_sample, channel)
    xai = list(xai_indices or [])
    free = list(dict.fromkeys(free_indices or []))
    lower, upper = _make_box(input_sample, eps, means, stddev)

    while True:
        remaining = n_dim - len(set(free)) - len(set(xai))
        if remaining <= 0:
            break
        cards = np.arange(1, remaining + 1, dtype=int)
        masks = free_at_once_k_features_l2(
            model=model,
            gt_label=gt_label,
            input_sample=input_sample,
            lower_bound_input=lower,
            upper_bound_input=upper,
            eps=eps,
            xai_indices=xai,
            free_indices=free,
            cardinality=cards,
            channel=channel,
            data_format=data_format,
            n_class=n_class,
            verbose=verbose,
            certificate_margin=certificate_margin,
        )
        if masks.size == 0 or np.max(np.sum(masks, axis=1)) == 0:
            break
        best = int(np.argmax(np.sum(masks, axis=1)))
        new_features = [
            int(j)
            for j in np.flatnonzero(masks[best])
            if int(j) not in free
        ]
        if not new_features:
            break
        free.extend(new_features)
        if not refining_domain:
            break

    singleton_added: list[int] = []
    if singleton_refinement and len(set(free)) + len(set(xai)) < n_dim:
        singleton_added = free_with_singleton_search_l2(
            model=model,
            gt_label=gt_label,
            input_sample=input_sample,
            lower_bound_input=lower,
            upper_bound_input=upper,
            eps=eps,
            free_indices=free,
            xai_indices=xai,
            channel=channel,
            data_format=data_format,
            n_class=n_class,
            certificate_margin=certificate_margin,
        )
    return free, singleton_added


def check_is_robust_l2(
    model: keras.models.Model,
    input_sample: np.ndarray,
    gt_label: int,
    eps: float,
    channel: int = 1,
    data_format: str = "channels_first",
    n_class: int = 10,
    certificate_margin: float = 1e-6,
) -> bool:
    """Certify the full L2 ball by setting k to all feature groups."""
    n_dim = _validate_layout(input_sample, channel)
    lower, upper = _make_box(input_sample, eps)
    _, _, upper_out, _ = get_features_batch_l2(
        model=model,
        gt_label=gt_label,
        input_sample=input_sample,
        lower_bound_input=lower,
        upper_bound_input=upper,
        xai_indices=[],
        free_indices=[],
        cardinality=np.asarray([n_dim], dtype=int),
        eps=eps,
        channel=channel,
        data_format=data_format,
        n_class=n_class,
        batch_size=1,
    )
    return bool(np.max(upper_out[0]) <= -certificate_margin)