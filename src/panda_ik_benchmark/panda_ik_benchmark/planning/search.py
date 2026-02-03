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


@dataclass(frozen=True)
class PrefixPathResult:
    """Prefix-optimal result for reaching p_i (i>=1).

    This is the DP equivalent of "exhaustive enumeration" for each prefix:
      - s1 = min_{q in IK(p1)} time(p0->q)
      - s2 = min_{q1 in IK(p1), q2 in IK(p2)} time(p0->q1) + time(q1->q2)
      - ...

    The best path is represented by the chosen IK solution indices (0-based)
    for each waypoint p1..p_i.
    """

    to_point_index_1based: int
    best_time_s: float
    solution_indices_0based: Sequence[int]


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
    the explicit exponential explosion.
    """
    # Keep old API but implement through the prefix DP (s_n is the global optimum).
    res, _prefix = global_optimal_prefix_dp(
        start_q=start_q,
        solutions_by_point=solutions_by_point,
        requested_per_point=requested_per_point,
        time_model=time_model,
    )
    return res


def global_optimal_prefix_dp(
    *,
    start_q: Sequence[float],
    solutions_by_point: Sequence[Sequence[IKSolution]],
    requested_per_point: int,
    time_model: SegmentTimeModel,
    block_size: int = 128,
) -> Tuple[PathResult, Sequence[PrefixPathResult]]:
    """Prefix-optimal dynamic programming with optional blocking.

    Returns
    -------
    (final_path_result, prefix_results)

    - final_path_result corresponds to s_n (best path to p_n).
    - prefix_results[i-1] corresponds to s_i (best path to p_i).

    Notes
    -----
    - The DP graph is a layered DAG: each IK solution at p_i is a node.
    - Edge costs are segment times between consecutive waypoints (rest-to-rest).
    - Complexity is O(sum_i K_{i-1}*K_i). For K up to 3000, blocking keeps peak
      memory manageable while remaining vectorized.
    """

    # Guardrail: MoveIt TOTG is extremely expensive per edge (it constructs a trajectory
    # and runs time-parameterization). DP needs *many* edges, so TOTG quickly becomes
    # intractable for large IK sets (e.g. 3000 solutions -> 9M edges per segment).
    #
    # For small problems users may still explicitly want TOTG, so only block it once
    # the theoretical edge count exceeds a conservative threshold.
    if str(time_model.info.effective) == "totg":
        edge_est = 0
        for li in range(1, len(solutions_by_point)):
            edge_est += int(len(solutions_by_point[li - 1])) * int(len(solutions_by_point[li]))
        # if edge_est > 250_000:
        #     raise RuntimeError(
        #         "time_model=totg is too slow for DP at this scale "
        #         f"(estimated edges={edge_est}). Use --time-model trapezoid (recommended)."
        #     )

    q0 = np.asarray(start_q, dtype=float)

    layers = [list(layer) for layer in solutions_by_point]
    if any(len(layer) == 0 for layer in layers):
        bad = [i + 1 for i, layer in enumerate(layers) if len(layer) == 0]
        raise RuntimeError(f"Some points have 0 IK solutions: {bad}")

    # Convert to numpy arrays
    Q = [np.asarray([s.joint_positions for s in layer], dtype=float) for layer in layers]
    dof = int(Q[0].shape[1])

    # dp for layer 0 (p1): dp1[j] = cost(p0 -> p1_j)
    t0 = time_model.segment_time_matrix_s(q0.reshape((1, dof)), Q[0])  # (1, K1)
    dp = t0.reshape((-1,))
    parents: List[np.ndarray] = [np.full((int(dp.shape[0]),), -1, dtype=int)]

    best_idx_per_layer: List[int] = [int(np.argmin(dp))]
    best_time_per_layer: List[float] = [float(dp[best_idx_per_layer[0]])]

    # iterate layers with blocking along the *current* layer to cap peak memory
    block_size = max(1, int(block_size))

    for li in range(1, len(Q)):
        prev = Q[li - 1]  # (Kprev, dof)
        curr = Q[li]      # (Kcurr, dof)
        Kprev = int(prev.shape[0])
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
            # gather best costs per column
            cols = np.arange(int(totals.shape[1]), dtype=int)
            dp_block = totals[parent_block, cols]

            dp_new[j0:j1] = dp_block
            parent[j0:j1] = parent_block

        parents.append(parent)
        dp = dp_new

        best_idx = int(np.argmin(dp))
        best_idx_per_layer.append(best_idx)
        best_time_per_layer.append(float(dp[best_idx]))

    # Build prefix results (s1..s_n)
    prefix_results: List[PrefixPathResult] = []
    for li in range(len(Q)):
        best_idx = int(best_idx_per_layer[li])

        # backtrack indices for this prefix
        idxs = [0] * (li + 1)
        idxs[li] = best_idx
        for k in range(li, 0, -1):
            idxs[k - 1] = int(parents[k][idxs[k]])

        prefix_results.append(
            PrefixPathResult(
                to_point_index_1based=int(li + 1),
                best_time_s=float(best_time_per_layer[li]),
                solution_indices_0based=tuple(int(x) for x in idxs),
            )
        )

    # Build PathResult for the final layer (s_n)
    final_idxs = list(prefix_results[-1].solution_indices_0based)
    segments: List[SegmentResult] = []
    current_q = q0.copy()
    for i, (layer, chosen_idx) in enumerate(zip(layers, final_idxs), start=1):
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

    res = PathResult(
        method="dp_global_optimal",
        segments=segments,
        total_time_s=float(prefix_results[-1].best_time_s),
        total_paths_theoretical=int(total_paths),
        time_model=tm_copy,
    )

    return res, tuple(prefix_results)
