from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from ..types import IKSolution
from .time_metric import SegmentTimeModel, TimeModelInfo


def _optimal_path_indices_dp(
    *,
    start_q: Sequence[float],
    layers: Sequence[Sequence[IKSolution]],
    time_model: SegmentTimeModel,
    block_size: int = 128,
) -> Tuple[Tuple[int, ...], float]:
    """Solve a small layered shortest-path problem and return the best indices.

    Parameters
    ----------
    start_q:
        Fixed start joint configuration (e.g. the already-committed IK of p{i-1}).
    layers:
        IK solution layers for consecutive points (e.g. [IK(p_i), IK(p_{i+1}), ...]).

    Returns
    -------
    (indices, best_time_s)
        - indices: chosen IK indices (0-based) for each layer.
        - best_time_s: minimal total time from start_q to the last layer, following these indices.

    Notes
    -----
    This is essentially the same layered-DAG shortest-path DP as the global solver,
    but scoped to a small window, and it only returns the best full-window path.
    """

    layers = [list(layer) for layer in layers]
    if len(layers) == 0:
        return tuple(), 0.0
    if any(len(layer) == 0 for layer in layers):
        bad = [i for i, layer in enumerate(layers) if len(layer) == 0]
        raise RuntimeError(f"Some window layers have 0 IK solutions: {bad}")

    q0 = np.asarray(start_q, dtype=float)

    # Convert to numpy arrays
    Q = [np.asarray([s.joint_positions for s in layer], dtype=float) for layer in layers]
    dof = int(Q[0].shape[1])

    # dp for layer 0: dp0[j] = cost(start -> layer0_j)
    t0 = time_model.segment_time_matrix_s(q0.reshape((1, dof)), Q[0])  # (1, K0)
    dp = t0.reshape((-1,))
    parents: List[np.ndarray] = [np.full((int(dp.shape[0]),), -1, dtype=int)]

    block_size = max(1, int(block_size))

    for li in range(1, len(Q)):
        prev = Q[li - 1]
        curr = Q[li]
        Kprev = int(prev.shape[0])
        Kcurr = int(curr.shape[0])

        dp_new = np.empty((Kcurr,), dtype=float)
        parent = np.empty((Kcurr,), dtype=int)

        for j0 in range(0, Kcurr, block_size):
            j1 = min(Kcurr, j0 + block_size)
            curr_block = curr[j0:j1]

            tmat = time_model.segment_time_matrix_s(prev, curr_block)  # (Kprev, B)
            totals = dp.reshape((-1, 1)) + tmat

            parent_block = np.argmin(totals, axis=0).astype(int)
            cols = np.arange(int(totals.shape[1]), dtype=int)
            dp_block = totals[parent_block, cols]

            dp_new[j0:j1] = dp_block
            parent[j0:j1] = parent_block

        parents.append(parent)
        dp = dp_new

    best_last = int(np.argmin(dp))
    best_time = float(dp[best_last])

    # backtrack
    idxs = [0] * len(Q)
    idxs[-1] = best_last
    for k in range(len(Q) - 1, 0, -1):
        idxs[k - 1] = int(parents[k][idxs[k]])

    return tuple(int(x) for x in idxs), best_time


@dataclass(frozen=True)
class SegmentResult:
    seg_idx_1based: int
    from_label: str
    to_label: str
    found: int
    requested: int
    best_solution: IKSolution
    best_time_s: float


@dataclass(frozen=True)
class PathResult:
    method: str
    segments: Sequence[SegmentResult]
    total_time_s: float
    total_paths_theoretical: int
    time_model: TimeModelInfo


def greedy_path(
    *,
    start_q: Sequence[float],
    solutions_by_point: Sequence[Sequence[IKSolution]],
    requested_per_point: int,
    time_model: SegmentTimeModel,
) -> PathResult:
    """
    Traditional greedy baseline:
      p0 -> choose best IK in p1 by minimal segment time,
      then from that IK choose best IK in p2, and so on.
    """
    current_q = np.asarray(start_q, dtype=float)

    segments: List[SegmentResult] = []
    total = 0.0

    for i, sols in enumerate(solutions_by_point, start=1):
        sols = list(sols)
        if len(sols) == 0:
            raise RuntimeError(f"Point p{i} has 0 IK solutions; greedy cannot continue.")

        times = []
        for s in sols:
            t = time_model.segment_time_s(current_q, s.joint_positions)
            times.append(t)

        j_best = int(np.argmin(np.asarray(times, dtype=float)))
        best_sol = sols[j_best]
        best_t = float(times[j_best])

        segments.append(
            SegmentResult(
                seg_idx_1based=i,
                from_label=f"p{i-1}",
                to_label=f"p{i}",
                found=len(sols),
                requested=int(requested_per_point),
                best_solution=best_sol,
                best_time_s=best_t,
            )
        )
        total += best_t
        current_q = np.asarray(best_sol.joint_positions, dtype=float)

    total_paths = 1
    for sols in solutions_by_point:
        total_paths *= max(1, int(len(sols)))

    tm = time_model.info
    tm_copy = TimeModelInfo(
        requested=str(tm.requested),
        effective=str(tm.effective),
        note=str(tm.note),
        totg_available=bool(tm.totg_available),
        totg_failures=int(tm.totg_failures),
    )

    return PathResult(
        method="greedy",
        segments=segments,
        total_time_s=float(total),
        total_paths_theoretical=int(total_paths),
        time_model=tm_copy,
    )


def window_path_receding_horizon(
    *,
    start_q: Sequence[float],
    solutions_by_point: Sequence[Sequence[IKSolution]],
    requested_per_point: int,
    time_model: SegmentTimeModel,
    window_size: int,
    block_size: int = 128,
) -> PathResult:
    """Sliding-window lookahead (receding horizon) path selection.

    At each point p_i (1-based), the IK for p_i is chosen by solving a *local* DP
    problem over a window of future waypoints:

      start = already committed IK of p_{i-1} (or p0 for i=1)
      window = [p_i, p_{i+1}, ..., p_{i+L-1}] where L = min(window_size, n-i+1)

    We find the shortest-time path from the start configuration to p_{i+L-1}
    through this window, then commit only the first decision (IK at p_i).

    This matches the user's spec:
      - if there are fewer remaining points near the end, the window shrinks.
      - the very last point p_n only considers the single edge p_{n-1}->p_n.
    """

    layers_all = [list(layer) for layer in solutions_by_point]
    if any(len(layer) == 0 for layer in layers_all):
        bad = [i + 1 for i, layer in enumerate(layers_all) if len(layer) == 0]
        raise RuntimeError(f"Some points have 0 IK solutions: {bad}")

    n = int(len(layers_all))
    w = int(window_size)
    if w < 1:
        w = 1

    current_q = np.asarray(start_q, dtype=float)
    segments: List[SegmentResult] = []
    total = 0.0

    for i0 in range(n):
        # i0 is 0-based for p_{i0+1}
        remain = n - i0
        L = min(w, remain)

        window_layers = layers_all[i0 : i0 + L]
        idxs, _best_window_time = _optimal_path_indices_dp(
            start_q=current_q,
            layers=window_layers,
            time_model=time_model,
            block_size=block_size,
        )

        chosen_idx = int(idxs[0])
        chosen_sol = window_layers[0][chosen_idx]

        dt = float(time_model.segment_time_s(current_q, chosen_sol.joint_positions))
        segments.append(
            SegmentResult(
                seg_idx_1based=i0 + 1,
                from_label=f"p{i0}",
                to_label=f"p{i0 + 1}",
                found=len(window_layers[0]),
                requested=int(requested_per_point),
                best_solution=chosen_sol,
                best_time_s=dt,
            )
        )
        total += dt
        current_q = np.asarray(chosen_sol.joint_positions, dtype=float)

    total_paths = 1
    for layer in layers_all:
        total_paths *= max(1, int(len(layer)))

    tm = time_model.info
    tm_copy = TimeModelInfo(
        requested=str(tm.requested),
        effective=str(tm.effective),
        note=str(tm.note),
        totg_available=bool(tm.totg_available),
        totg_failures=int(tm.totg_failures),
    )

    return PathResult(
        method="window",
        segments=segments,
        total_time_s=float(total),
        total_paths_theoretical=int(total_paths),
        time_model=tm_copy,
    )


def global_optimal_path(
    *,
    start_q: Sequence[float],
    solutions_by_point: Sequence[Sequence[IKSolution]],
    requested_per_point: int,
    time_model: SegmentTimeModel,
    block_size: int = 128,
) -> PathResult:
    """Global method: enumerate all full paths (conceptually) and pick the shortest.

    The mathematical definition matches the user's "乘法原理" description:

      min_{q1 in IK(p1), ..., qn in IK(pn)} [ time(p0->q1) + time(q1->q2) + ... + time(q{n-1}->qn) ]

    Implementation note
    -------------------
    Explicit full enumeration is exponential (|IK(p1)|*...*|IK(pn)|). We compute the
    same result using a shortest-path DP on a layered DAG, and we only backtrack once
    for the final path (0-1-2-...-n), without producing prefix-optimal outputs.
    """

    q0 = np.asarray(start_q, dtype=float)
    layers = [list(layer) for layer in solutions_by_point]
    if any(len(layer) == 0 for layer in layers):
        bad = [i + 1 for i, layer in enumerate(layers) if len(layer) == 0]
        raise RuntimeError(f"Some points have 0 IK solutions: {bad}")

    # Convert to numpy arrays
    Q = [np.asarray([s.joint_positions for s in layer], dtype=float) for layer in layers]
    dof = int(Q[0].shape[1])

    # dp for layer 0 (p1): dp[j] = cost(p0 -> p1_j)
    t0 = time_model.segment_time_matrix_s(q0.reshape((1, dof)), Q[0])  # (1, K1)
    dp = t0.reshape((-1,))
    parents: List[np.ndarray] = [np.full((int(dp.shape[0]),), -1, dtype=int)]

    block_size = max(1, int(block_size))

    # iterate layers with blocking along the *current* layer to cap peak memory
    for li in range(1, len(Q)):
        prev = Q[li - 1]  # (Kprev, dof)
        curr = Q[li]      # (Kcurr, dof)
        Kcurr = int(curr.shape[0])

        dp_new = np.empty((Kcurr,), dtype=float)
        parent = np.empty((Kcurr,), dtype=int)

        for j0 in range(0, Kcurr, block_size):
            j1 = min(Kcurr, j0 + block_size)
            curr_block = curr[j0:j1]

            # cost matrix for the block: (Kprev, B)
            tmat = time_model.segment_time_matrix_s(prev, curr_block)

            # totals: (Kprev, B)
            totals = dp.reshape((-1, 1)) + tmat
            parent_block = np.argmin(totals, axis=0).astype(int)  # (B,)
            cols = np.arange(int(totals.shape[1]), dtype=int)
            dp_block = totals[parent_block, cols]

            dp_new[j0:j1] = dp_block
            parent[j0:j1] = parent_block

        parents.append(parent)
        dp = dp_new

    best_last = int(np.argmin(dp))

    # Backtrack ONLY the final full path indices (p1..pn)
    idxs = [0] * len(Q)
    idxs[-1] = best_last
    for k in range(len(Q) - 1, 0, -1):
        idxs[k - 1] = int(parents[k][idxs[k]])

    # Build PathResult
    segments: List[SegmentResult] = []
    current_q = q0.copy()
    total = 0.0
    for i, (layer, chosen_idx) in enumerate(zip(layers, idxs), start=1):
        sol = layer[int(chosen_idx)]
        t = float(time_model.segment_time_s(current_q, sol.joint_positions))
        segments.append(
            SegmentResult(
                seg_idx_1based=i,
                from_label=f"p{i-1}",
                to_label=f"p{i}",
                found=len(layer),
                requested=int(requested_per_point),
                best_solution=sol,
                best_time_s=t,
            )
        )
        total += t
        current_q = np.asarray(sol.joint_positions, dtype=float)

    total_paths = 1
    for layer in layers:
        total_paths *= max(1, int(len(layer)))

    tm = time_model.info
    tm_copy = TimeModelInfo(
        requested=str(tm.requested),
        effective=str(tm.effective),
        note=str(tm.note),
        totg_available=bool(tm.totg_available),
        totg_failures=int(tm.totg_failures),
    )

    return PathResult(
        method="global",
        segments=segments,
        total_time_s=float(total),
        total_paths_theoretical=int(total_paths),
        time_model=tm_copy,
    )
