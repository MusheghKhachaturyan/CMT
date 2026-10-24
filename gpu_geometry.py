"""Float64 ordered-pair pose estimation, with exact (never tiled) medians."""

import numpy as np


def workspace_bytes(keypoints, scale=True, rotation=True):
    k = len(keypoints)
    _, counts = np.unique(keypoints[:, 2].astype(int), return_counts=True)
    pairs = k * k - sum(int(c) ** 2 for c in counts)
    if not pairs or not (scale or rotation):
        return 0
    # Conservative live-array bound: mask, indices, gathered coordinates and
    # references, float64 arithmetic, compacted values and median sort scratch
    # (measured peak 73 bytes per pair; 112 budgeted).
    return 65536 + k * 64 + k * k * 16 + pairs * 112


def pair_pose(backend, model, sorted_keypoints, estimate_scale, estimate_rotation):
    host_classes = sorted_keypoints[:, 2]
    if (not np.isfinite(host_classes).all() or np.any(host_classes != host_classes.astype(int))
            or np.any(host_classes < 1) or np.any(host_classes > model.selected_count)):
        raise ValueError("Keypoint classes must index the selected model (1..N)")
    if len(np.unique(host_classes.astype(int))) < 2:
        return np.array([np.nan, np.nan])
    if not (estimate_scale or estimate_rotation):
        return np.array([1., 0.])
    cp = backend.cp
    with backend.device:
        points = backend.upload(np.asarray(sorted_keypoints, dtype=np.float64))
        classes = points[:, 2].astype(cp.int32) - 1
        i, j = cp.nonzero(classes[:, None] != classes[None, :])
        if not i.size:
            return np.array([np.nan, np.nan])
        scale = cp.asarray(1., dtype=cp.float64)
        rotation = cp.asarray(0., dtype=cp.float64)
        if estimate_scale:
            delta = points[i, :2] - points[j, :2]
            distances = cp.sqrt((delta ** 2).sum(axis=1))
            reference = model.distances[classes[i], classes[j]]
            valid = reference > 0
            ratios = distances[valid] / reference[valid]
            scale = cp.median(ratios) if ratios.size else scale
            del delta, distances, reference, valid, ratios
        if estimate_rotation:
            delta = points[j, :2] - points[i, :2]
            angles = cp.arctan2(delta[:, 1], delta[:, 0]) - model.angles[classes[i], classes[j]]
            angles = cp.where(cp.abs(angles) > np.pi, angles - cp.sign(angles) * (2 * np.pi), angles)
            rotation = cp.median(angles)
        return backend.download(cp.stack((scale, rotation)))
