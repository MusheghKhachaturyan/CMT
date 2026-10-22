"""Bounded packed-bit Hamming matching and fused structural scoring."""

import numpy as np

from backend import BackendError, validate_descriptors


POPCOUNT = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def tile_shape(queries, database, budget):
    # Covers two byte tensors, reductions, float64 XY/sqrt scoring, masks,
    # indices, reduction scratch and per-query candidate/transfer storage.
    usable = budget - queries * 192 - 65536
    if usable < 256:
        raise BackendError("GPU matching workspace too small for query buffers; increase --gpu-memory-mb")
    columns = min(database, 256, max(1, usable // 256))
    rows = min(queries, 128, max(1, usable // (columns * 256)))
    return rows, columns


def hamming_tile(cp, queries, database, lookup):
    xor = cp.bitwise_xor(queries[:, None, :], database[None, :, :])
    return cp.sum(lookup[xor], axis=2, dtype=cp.uint16)


def _two_best(cp, distances, indices):
    """Two minima, preserving index ordering for otherwise irrelevant ties."""
    rows = cp.arange(len(distances))
    first = cp.argmin(distances, axis=1)
    best = distances[rows, first]
    best_index = indices[rows, first]
    distances[rows, first] = 65535
    second = cp.argmin(distances, axis=1)
    return cp.stack((best, distances[rows, second]), axis=1), cp.stack((best_index, indices[rows, second]), axis=1)


def _merge(cp, old_dist, old_index, distances, offset):
    indices = cp.broadcast_to(cp.arange(offset, offset + distances.shape[1], dtype=cp.int32), distances.shape)
    if distances.shape[1] == 1:
        dist = cp.concatenate((distances, cp.full_like(distances, 65535)), axis=1)
        index = cp.concatenate((indices, cp.full_like(indices, -1)), axis=1)
    else:
        dist, index = _two_best(cp, distances, indices)
    return _two_best(cp, cp.concatenate((old_dist, dist), axis=1), cp.concatenate((old_index, index), axis=1))


def _accepted(cp, distances, indices, classes, confidence, ratio):
    d = distances.astype(cp.float64)
    # where evaluates both branches: substitute 1 in the denominator first.
    denominator = cp.where(d[:, 1] != 0, d[:, 1], 1.)
    ratios = cp.where(d[:, 1] != 0, d[:, 0] / denominator, 1.)
    winner = classes[cp.maximum(indices[:, 0], 0)]
    valid = (indices[:, 0] >= 0) & (indices[:, 1] >= 0) & (ratios < ratio) & (1 - d[:, 0] / 512 > confidence) & (winner != 0)
    return cp.where(valid, winner, 0).astype(cp.int32)


def match_candidates(backend, model, features, positions, center, transformed_springs,
                     outlier_threshold, confidence_threshold, ratio_threshold):
    features = validate_descriptors(features)
    q = len(features)
    events = np.zeros((q, 2), dtype=np.int32)
    d = len(model.descriptors)
    if not q or d < 2:
        return events
    cp = backend.cp
    structural = not np.isnan(center).any() and model.selected_count >= 2
    budget = min(backend.workspace_bytes, max(0, backend.available_bytes() - 1024 ** 2))
    if structural:
        budget -= model.selected_count * 16 + 16
    qr, dc = tile_shape(q, d, budget)
    with backend.device:
        if not hasattr(backend, "popcount"):
            backend.popcount = backend.upload(POPCOUNT)
        lookup = backend.popcount
        # Upload descriptors/positions once; never upload images.
        query = backend.upload(features)
        xy = backend.upload(np.asarray(positions, dtype=np.float64)) if structural else None
        springs = backend.upload(np.asarray(transformed_springs, dtype=np.float64)) if structural else None
        center_gpu = backend.upload(np.asarray(center, dtype=np.float64)) if structural else None
        output = cp.zeros((q, 2), dtype=cp.int32)
        for start in range(0, q, qr):
            end = min(start + qr, q)
            best = cp.full((end - start, 2), 65535, dtype=cp.uint16)
            index = cp.full((end - start, 2), -1, dtype=cp.int32)
            sbest, sindex = best.copy(), index.copy()
            for col in range(0, d, dc):
                stop = min(col + dc, d)
                distances = hamming_tile(cp, query[start:end], model.descriptors[col:stop], lookup)
                # Score selected suffix before global reductions mutate the tile.
                first_selected = max(col, model.selected_offset)
                if structural and first_selected < stop:
                    local = first_selected - model.selected_offset
                    delta = springs[None, local:stop - model.selected_offset, :] - (xy[start:end] - center_gpu)[:, None, :]
                    displacement = cp.sqrt((delta ** 2).sum(axis=2))
                    effective = cp.where(displacement < outlier_threshold,
                                         distances[:, first_selected - col:], 512).astype(cp.uint16)
                    sbest, sindex = _merge(cp, sbest, sindex, effective, first_selected)
                best, index = _merge(cp, best, index, distances, col)
            output[start:end, 0] = _accepted(cp, best, index, model.classes, confidence_threshold, ratio_threshold)
            if structural:
                output[start:end, 1] = _accepted(cp, sbest, sindex, model.classes, confidence_threshold, ratio_threshold)
        return backend.download(output)


def replay_events(positions, events):
    """Literal legacy event replay; deliberately retains duplicate global rows."""
    active = np.zeros((0, 3))
    for location, (global_class, structural_class) in zip(positions, events):
        if global_class:
            active = np.append(active, [np.append(location, global_class)], axis=0)
        if structural_class:
            if active.size:
                active = np.delete(active, np.nonzero(active[:, 2] == structural_class), axis=0)
            active = np.append(active, [np.append(location, structural_class)], axis=0)
    return active
