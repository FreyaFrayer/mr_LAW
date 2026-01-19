from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from ..types import IKSolution
from .time_metric import SegmentTimeModel, TimeModelInfo


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


def global_optimal_path_dp(
    *,
    start_q: Sequence[float],
    solutions_by_point: Sequence[Sequence[IKSolution]],
    requested_per_point: int,
    time_model: SegmentTimeModel,
) -> PathResult:
    """
    Globally optimal path using dynamic programming (shortest path in a layered DAG).

    This is mathematically equivalent to enumerating all IK combinations, but avoids
    the 200^n explicit explosion. Complexity is O(n * K^2) with K≈200.
    """
    q0 = np.asarray(start_q, dtype=float)

    layers = [list(layer) for layer in solutions_by_point]
    if any(len(layer) == 0 for layer in layers):
        bad = [i + 1 for i, layer in enumerate(layers) if len(layer) == 0]
        raise RuntimeError(f"Some points have 0 IK solutions: {bad}")

    # Convert to numpy arrays
    Q = [np.asarray([s.joint_positions for s in layer], dtype=float) for layer in layers]
    K = [int(q.shape[0]) for q in Q]
    dof = int(Q[0].shape[1])

    # dp for layer 0 (p1): dp1[j] = cost(p0 -> p1_j)
    t0 = time_model.segment_time_matrix_s(q0.reshape((1, dof)), Q[0])  # (1, K1)
    dp = t0.reshape((-1,))
    parents: List[np.ndarray] = [np.full((K[0],), -1, dtype=int)]

    # iterate layers
    for li in range(1, len(Q)):
        prev = Q[li - 1]  # (Kprev, dof)
        curr = Q[li]      # (Kcurr, dof)

        # cost matrix: (Kprev, Kcurr)
        tmat = time_model.segment_time_matrix_s(prev, curr)

        # dp_new[j] = min_k dp[k] + tmat[k, j]
        totals = dp.reshape((-1, 1)) + tmat
        parent = np.argmin(totals, axis=0).astype(int)     # (Kcurr,)
        dp_new = totals[parent, np.arange(totals.shape[1])]  # (Kcurr,)

        parents.append(parent)
        dp = dp_new

    # choose best final solution
    best_last = int(np.argmin(dp))
    best_total = float(dp[best_last])

    # backtrack indices
    idxs = [0] * len(Q)
    idxs[-1] = best_last
    for li in range(len(Q) - 1, 0, -1):
        idxs[li - 1] = int(parents[li][idxs[li]])

    # build segments with chosen solutions
    segments: List[SegmentResult] = []
    current_q = q0.copy()
    for i, (layer, chosen_idx) in enumerate(zip(layers, idxs), start=1):
        sol = layer[int(chosen_idx)]
        t = time_model.segment_time_s(current_q, sol.joint_positions)
        segments.append(
            SegmentResult(
                seg_idx_1based=i,
                from_label=f"p{i-1}",
                to_label=f"p{i}",
                found=len(layer),
                requested=int(requested_per_point),
                best_solution=sol,
                best_time_s=float(t),
            )
        )
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
        method="dp_global_optimal",
        segments=segments,
        total_time_s=float(best_total),
        total_paths_theoretical=int(total_paths),
        time_model=tm_copy,
    )
