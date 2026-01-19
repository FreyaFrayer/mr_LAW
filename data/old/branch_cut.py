import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class IKSolution:
    attempt: int
    yaw: float
    q: np.ndarray  # shape (dof,)
    branch: Tuple[int, int, int]  # (sign(q2), sign(q3), sign(q5))


# -----------------------------
# Utilities
# -----------------------------
def sign_with_eps(x: float, eps: float = 1e-9) -> int:
    if x >= eps:
        return 1
    if x <= -eps:
        return -1
    return 0


def branch_signature(q: np.ndarray, eps: float = 1e-9) -> Tuple[int, int, int]:
    """
    A cheap discrete signature to separate IK modes.
    For Panda-like 7DoF arms, (q2, q3, q5) signs often capture elbow/wrist mode flips.
    """
    return (
        sign_with_eps(float(q[1]), eps),  # q2
        sign_with_eps(float(q[2]), eps),  # q3
        sign_with_eps(float(q[4]), eps),  # q5
    )


def weighted_linf(q1: np.ndarray, q2: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Cheap surrogate distance. L_infinity is aligned with "bottleneck joint" intuition.
    If weights are provided, distance is max_i |dq_i| * w_i .
    """
    dq = np.abs(q2 - q1)
    if weights is not None:
        dq = dq * weights
    return float(np.max(dq))


def min_dist_to_set(q: np.ndarray, Q: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    For a single q, compute min over rows of Q of weighted L_infinity distance.
    Complexity O(|Q| * dof), but here |Q|<=200 so it's cheap.
    """
    if Q.size == 0:
        return float("inf")
    dq = np.abs(Q - q[None, :])  # (n, dof)
    if weights is not None:
        dq = dq * weights[None, :]
    return float(np.min(np.max(dq, axis=1)))


def load_ik_solutions(json_path: str, dof: int = 7) -> List[IKSolution]:
    with open(json_path, "r") as f:
        data = json.load(f)

    sols: List[IKSolution] = []
    for s in data["solutions"]:
        q = np.array(s["joint_positions"], dtype=float)[:dof]
        yaw = float(s.get("sampled_yaw_rad", 0.0))
        attempt = int(s["attempt"])
        sols.append(IKSolution(attempt=attempt, yaw=yaw, q=q, branch=branch_signature(q)))
    return sols


def group_by_branch(solutions: List[IKSolution]) -> Dict[Tuple[int, int, int], List[IKSolution]]:
    groups: Dict[Tuple[int, int, int], List[IKSolution]] = {}
    for sol in solutions:
        groups.setdefault(sol.branch, []).append(sol)
    return groups


def branch_score(
    B_branch: List[IKSolution],
    C_branch: List[IKSolution],
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Score a branch by how "compatible" B and C are in joint space.
    Lower is better. Use median of min-distances from each B to the C-set.
    """
    if len(B_branch) == 0 or len(C_branch) == 0:
        return float("inf")
    QC = np.stack([c.q for c in C_branch], axis=0)
    dmins = [min_dist_to_set(b.q, QC, weights=weights) for b in B_branch]
    return float(np.median(dmins))


def choose_redundant_joint_index(B_branch: List[IKSolution], C_branch: List[IKSolution]) -> int:
    """
    Heuristic: the joint with the largest variance in this branch is treated as the redundancy axis.
    For Panda, it is often joint7 (index 6).
    """
    Q = np.vstack([np.stack([b.q for b in B_branch]), np.stack([c.q for c in C_branch])])
    var = np.var(Q, axis=0)
    return int(np.argmax(var))


def select_k_by_redundancy_binning(
    sols: List[IKSolution],
    other_set_Q: np.ndarray,
    K: int,
    redundancy_index: int,
    weights: Optional[np.ndarray] = None,
) -> List[IKSolution]:
    """
    Reduce a branch solution set to K representatives:
    - Bin along the redundancy axis into K bins
    - In each bin, keep the solution with the smallest min-dist to the other endpoint set
    - Fill remaining slots by globally best min-dist (if some bins empty)

    This keeps both:
    (1) coverage over redundancy (avoid missing a good "arm angle")
    (2) preference for B-C continuity (cheap proxy for B->C time)
    """
    if len(sols) <= K:
        return sols

    Q = np.stack([s.q for s in sols], axis=0)
    qualities = np.array([min_dist_to_set(s.q, other_set_Q, weights=weights) for s in sols], dtype=float)

    r = Q[:, redundancy_index]
    r_min, r_max = float(r.min()), float(r.max())

    selected_idx = set()

    # 1) Bin representatives
    if r_max > r_min:
        edges = np.linspace(r_min, r_max, K + 1)
        for bi in range(K):
            if bi < K - 1:
                mask = (r >= edges[bi]) & (r < edges[bi + 1])
            else:
                mask = (r >= edges[bi]) & (r <= edges[bi + 1])
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue
            best = int(idx[np.argmin(qualities[idx])])
            selected_idx.add(best)
            if len(selected_idx) >= K:
                break

    # 2) Fill by best quality if bins were empty
    order = np.argsort(qualities)
    for i in order:
        if len(selected_idx) >= K:
            break
        selected_idx.add(int(i))

    selected = [sols[i] for i in sorted(selected_idx, key=lambda i: qualities[i])]
    return selected[:K]


def prune_to_20x20(
    path_b_json: str,
    path_c_json: str,
    K_B: int = 20,
    K_C: int = 20,
    keep_top_branches: int = 1,
    weights: Optional[np.ndarray] = None,
) -> Tuple[List[IKSolution], List[IKSolution], List[Tuple[float, Tuple[int, int, int]]]]:
    """
    Main pruning pipeline:
    1) Split solutions by branch (IK mode)
    2) Score branches by B-C compatibility, keep top branches
    3) In each kept branch, reduce B and C to K representatives using redundancy-axis binning
    """
    B_all = load_ik_solutions(path_b_json)
    C_all = load_ik_solutions(path_c_json)

    B_groups = group_by_branch(B_all)
    C_groups = group_by_branch(C_all)

    common_branches = sorted(set(B_groups.keys()) & set(C_groups.keys()))
    scored: List[Tuple[float, Tuple[int, int, int]]] = []
    for br in common_branches:
        s = branch_score(B_groups[br], C_groups[br], weights=weights)
        scored.append((s, br))
    scored.sort(key=lambda x: x[0])

    kept_branches = [br for _, br in scored[:keep_top_branches]]

    B_keep: List[IKSolution] = []
    C_keep: List[IKSolution] = []

    # Allocate budget evenly across kept branches
    alloc_B = [K_B // keep_top_branches] * keep_top_branches
    alloc_C = [K_C // keep_top_branches] * keep_top_branches
    for i in range(K_B - sum(alloc_B)):
        alloc_B[i] += 1
    for i in range(K_C - sum(alloc_C)):
        alloc_C[i] += 1

    for br, kb, kc in zip(kept_branches, alloc_B, alloc_C):
        B_branch = B_groups[br]
        C_branch = C_groups[br]
        ridx = choose_redundant_joint_index(B_branch, C_branch)

        QB = np.stack([b.q for b in B_branch], axis=0)
        QC = np.stack([c.q for c in C_branch], axis=0)

        B_sel = select_k_by_redundancy_binning(B_branch, other_set_Q=QC, K=kb, redundancy_index=ridx, weights=weights)
        C_sel = select_k_by_redundancy_binning(C_branch, other_set_Q=QB, K=kc, redundancy_index=ridx, weights=weights)

        B_keep.extend(B_sel)
        C_keep.extend(C_sel)

    return B_keep, C_keep, scored


# -----------------------------
# Expensive evaluation interface
# -----------------------------
def time_lower_bound_vel(q1: np.ndarray, q2: np.ndarray, v_max: np.ndarray) -> float:
    """
    Admissible lower bound using joint velocity limits:
        t >= max_i |dq_i| / v_max_i
    If you have real joint limits from URDF/MoveIt, this bound is safe for pruning.
    """
    dq = np.abs(q2 - q1)
    return float(np.max(dq / v_max))


def search_best_pair(
    qA: np.ndarray,
    B_keep: List[IKSolution],
    C_keep: List[IKSolution],
    plan_time: Callable[[np.ndarray, np.ndarray], float],
    v_max: Optional[np.ndarray] = None,
) -> Tuple[Tuple[int, int], float]:
    """
    Evaluate only B_keep x C_keep (typically ~400 pairs).
    Optional: branch-and-bound pruning using a velocity-based lower bound.
    """
    best_total = float("inf")
    best_pair = (-1, -1)

    # Precompute t(A->B) (this is at most 20 calls)
    tAB: Dict[int, float] = {}
    for b in B_keep:
        tAB[b.attempt] = float(plan_time(qA, b.q))

    # Evaluate pairs
    for b in B_keep:
        for c in C_keep:
            if b.branch != c.branch:
                continue

            # Optional pruning: if even the LB cannot beat current best, skip expensive planning
            if v_max is not None:
                lb = tAB[b.attempt] + time_lower_bound_vel(b.q, c.q, v_max=v_max)
                if lb >= best_total:
                    continue

            tBC = float(plan_time(b.q, c.q))
            total = tAB[b.attempt] + tBC
            if total < best_total:
                best_total = total
                best_pair = (b.attempt, c.attempt)

    return best_pair, best_total


if __name__ == "__main__":
    # Example usage on your files:
    path_B = "panda_ik_solutions3.json"
    path_C = "panda_ik_solutions2.json"

    # If you want time-like weighting, you can set weights = 1 / v_max (roughly).
    # Here we keep it None (uniform) because it is only used for cheap pruning.
    B_keep, C_keep, scored = prune_to_20x20(
        path_b_json=path_B,
        path_c_json=path_C,
        K_B=20,
        K_C=20,
        keep_top_branches=1,
        weights=None,
    )

    print("Top branches by compatibility score (lower is better):")
    for s, br in scored[:4]:
        print(f"  branch={br}, score={s:.4f}")

    print("\nSelected B attempts:", [b.attempt for b in B_keep])
    print("Selected C attempts:", [c.attempt for c in C_keep])

    # To actually compute best pair, implement plan_time() and call search_best_pair().
    # def plan_time(q_start: np.ndarray, q_goal: np.ndarray) -> float:
    #     # TODO: call your MoveIt planner + TOTG, return trajectory duration (seconds)
    #     raise NotImplementedError
    #
    # qA = np.array([...], dtype=float)  # your start joint config
    # v_max = np.array([...], dtype=float)  # joint velocity limits from URDF/MoveIt
    # best_pair, best_total = search_best_pair(qA, B_keep, C_keep, plan_time, v_max=v_max)
    # print(best_pair, best_total)
