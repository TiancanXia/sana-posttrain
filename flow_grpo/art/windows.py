# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DiffusionART contributors
"""Prompt windows from DiffusionART-main/src/visual_rl/training/objectives.py.

The function body is retained unchanged; only its module imports are reduced.
"""

import torch


def prompt_disjoint_windows(groups, selected_count: int, updates: int = 2, *, order=None):
    """Split each complete prompt group, carrying every selected slot with its row.

    Group IDs identify rollout prompt instances, not prompt text. Callers compute
    guidance over complete groups before partitioning; inactive rows stay present.
    An optional row permutation controls order without changing group budgets.
    """
    if (
        groups.ndim != 1 or not groups.numel()
        or groups.dtype not in (torch.int32, torch.int64) or (groups < 0).any()
        or any(type(x) is not int or x <= 0 for x in (selected_count, updates))
    ):
        raise ValueError("prompt windows require nonempty integer groups and positive counts")
    ids = groups.detach().cpu().tolist()
    order = list(range(len(ids))) if order is None else list(order)
    if any(type(row) is not int for row in order) or sorted(order) != list(range(len(ids))):
        raise ValueError("order must be a permutation of all rollout rows")
    members = {}
    for row in order:
        members.setdefault(ids[row], []).append(row)
    if any(len(rows) % updates for rows in members.values()):
        raise ValueError("each prompt group must divide evenly into update windows")
    assignments = [0] * len(ids)
    for rows in members.values():
        size = len(rows) // updates
        for position, row in enumerate(rows):
            assignments[row] = position // size
    return tuple(
        tuple((row, slot) for row in order if assignments[row] == window
              for slot in range(selected_count))
        for window in range(updates)
    )
