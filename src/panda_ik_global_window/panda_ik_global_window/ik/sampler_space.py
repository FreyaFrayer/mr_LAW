from __future__ import annotations

import json
import math
import time

from dataclasses import dataclass

@dataclass
class DeterministicRNG:
    """Local deterministic RNG (no global random state).

    This is designed to make IK sampling reproducible across runs when `seed` is fixed.
    """

    seed: int

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(int(self.seed))

    def uniform01(self) -> float:
        return float(self._rng.random())

    def uniform(self, low: float, high: float) -> float:
        return float(self._rng.uniform(float(low), float(high)))

    def uniform_vec(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        # numpy supports vector low/high broadcast
        return np.asarray(self._rng.uniform(low, high), dtype=float)

    def normal_vec(self, n: int, *, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
        return np.asarray(self._rng.normal(float(mean), float(std), int(n)), dtype=float)

    def shuffle_inplace(self, xs: List[float]) -> None:
        if len(xs) <= 1:
            return
        order = self._rng.permutation(len(xs))
        xs[:] = [xs[i] for i in order]
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from geometry_msgs.msg import Pose

from moveit.core.robot_state import RobotState

from ..types import IKSolution, TargetPoint
from ..utils.robot import RobotContext


def quat_multiply(
    q1: Tuple[float, float, float, float],
    q2: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Hamilton product: q = q1 ⊗ q2, quaternion format (x, y, z, w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return (x, y, z, w)


def quat_normalize(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / n, y / n, z / n, w / n)


def quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    """Rotation around +Z axis by yaw (radians), quaternion (x,y,z,w)."""
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def build_pose(point: TargetPoint, quat_xyzw: Tuple[float, float, float, float]) -> Pose:
    pose = Pose()
    pose.position.x = float(point.x)
    pose.position.y = float(point.y)
    pose.position.z = float(point.z)
    pose.orientation.x = float(quat_xyzw[0])
    pose.orientation.y = float(quat_xyzw[1])
    pose.orientation.z = float(quat_xyzw[2])
    pose.orientation.w = float(quat_xyzw[3])
    return pose


def hash_joint_vector(q: np.ndarray, resolution: float) -> Tuple[int, ...]:
    """Quantize joint vector to create a stable uniqueness key."""
    return tuple(int(round(float(v) / resolution)) for v in q)


def clamp_to_group_bounds(q: np.ndarray, jmg) -> np.ndarray:
    """
    Clamp q to the active joint bounds of JointModelGroup (if available).
    If bounds unavailable / mismatch, return q unchanged.
    """
    try:
        bounds = list(jmg.active_joint_model_bounds)
    except Exception:
        return q

    if len(bounds) != int(q.shape[0]):
        return q

    qq = np.array(q, dtype=float, copy=True)
    for i, b in enumerate(bounds):
        try:
            if bool(getattr(b, "position_bounded")):
                lo = float(getattr(b, "min_position"))
                hi = float(getattr(b, "max_position"))
                if lo <= hi:
                    if qq[i] < lo:
                        qq[i] = lo
                    elif qq[i] > hi:
                        qq[i] = hi
        except Exception:
            pass
    return qq


def compute_nullspace_direction(state: RobotState, group: str, tip_link: str, rng: DeterministicRNG) -> Optional[np.ndarray]:
    """
    Compute one (randomized) nullspace direction in joint space from Jacobian J(q).

    For Panda arm (7-DoF), this is typically 1D nullspace.
    """
    ref = np.zeros(3, dtype=float)
    try:
        # Prefer explicit link_name overload if available
        J = state.get_jacobian(group, tip_link, ref, False)
    except TypeError:
        # Fallback: group + reference point overload
        try:
            J = state.get_jacobian(group, ref)
        except Exception:
            return None
    except Exception:
        return None

    if not isinstance(J, np.ndarray):
        J = np.array(J, dtype=float)
    if J.ndim != 2:
        return None

    dof = int(J.shape[1])
    if dof <= 0:
        return None

    try:
        _U, S, Vt = np.linalg.svd(J, full_matrices=True)
    except np.linalg.LinAlgError:
        return None

    tol = 1e-6
    rank = int(np.sum(S > tol))
    if rank >= dof:
        return None

    null_basis = Vt.T[:, rank:]  # (dof, null_dim)
    if null_basis.size == 0:
        return None

    weights = rng.normal_vec(int(null_basis.shape[1]), mean=0.0, std=1.0)
    n = null_basis @ weights
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-12:
        n = null_basis[:, -1]
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-12:
            return None

    # Canonicalize sign for reproducibility (nullspace vectors have sign ambiguity).
    try:
        k = int(np.argmax(np.abs(n)))
        if k >= 0 and float(n[k]) < 0.0:
            n = -n
    except Exception:
        pass

    return (n / n_norm).reshape((-1,))


def stratified_yaw(space_idx: int, num_spaces: int, yaw_range: float, rng: DeterministicRNG) -> float:
    """
    Stratified random sampling of yaw within [-yaw_range/2, +yaw_range/2].
    """
    if num_spaces <= 1:
        return (rng.uniform01() - 0.5) * float(yaw_range)

    bin_w = float(yaw_range) / float(num_spaces)
    return (-0.5 * float(yaw_range)) + (float(space_idx) + rng.uniform01()) * bin_w


def sample_ik_solutions(
    ctx: RobotContext,
    *,
    target_point: TargetPoint,
    nominal_tip_quat_xyzw: Tuple[float, float, float, float],
    named_start_for_seeding: str = "ready",
    num_solutions: int = 200,
    num_spaces: int = 20,
    max_attempts: int = 20000,
    ik_timeout_s: float = 0.05,
    yaw_range_rad: float = 2.0 * math.pi,
    nullspace_step: float = 0.20,
    nullspace_jitter: float = 0.02,
    uniq_resolution_rad: float = 1e-3,
    seed: int = 7,
) -> Dict:
    """
    Sample many IK solutions for one Cartesian point.

    This is adapted from the provided `ik_sampler_space.py` implementation,
    keeping the same "yaw spaces + nullspace exploration" idea.

    Returns
    -------
    A dict with:
      - meta: dict
      - solutions: List[dict]  (JSON-friendly)
      - solutions_obj: List[IKSolution]  (typed objects, convenient for search)
    """
    rng = DeterministicRNG(int(seed))

    # NOTE: This benchmark supports up to 3000 IK solutions per point.
    # Keep a hard cap to prevent accidental huge JSON / memory blowups.
    num_solutions = int(num_solutions)
    num_solutions = max(1, min(3000, num_solutions))

    num_spaces = int(num_spaces)
    num_spaces = max(1, min(num_solutions, num_spaces))

    nullspace_step = float(nullspace_step) if float(nullspace_step) > 1e-9 else 0.20
    nullspace_jitter = float(nullspace_jitter) if float(nullspace_jitter) >= 0.0 else 0.0

    solutions_per_space_target = int(math.ceil(float(num_solutions) / float(num_spaces)))
    solutions_per_space_target = max(1, solutions_per_space_target)

    robot_model = ctx.robot_model
    group = ctx.group
    tip_link = ctx.tip_link

    jmg = robot_model.get_joint_model_group(group)
    joint_names = list(ctx.joint_names)
    dof = len(joint_names)

    # Deterministic joint seed sampling (avoid MoveIt internal RNG).
    lows = np.array([jl.min_position for jl in ctx.joint_limits], dtype=float)
    highs = np.array([jl.max_position for jl in ctx.joint_limits], dtype=float)
    if lows.shape[0] != highs.shape[0] or int(lows.shape[0]) != dof:
        # Fallback to a safe range if limits are not aligned.
        lows = np.full((dof,), -math.pi, dtype=float)
        highs = np.full((dof), +math.pi, dtype=float)

    # Handle any degenerate bounds
    span = highs - lows
    span = np.where(span > 1e-12, span, 2.0 * math.pi)
    highs = lows + span

    q_nominal = quat_normalize(tuple(map(float, nominal_tip_quat_xyzw)))

    uniq_keys = set()
    solutions: List[Dict] = []
    solutions_obj: List[IKSolution] = []

    t0 = time.time()
    attempts = 0
    successes = 0

    space_attempt_budget = max(50, int(float(max_attempts) / float(num_spaces)))

    for space_idx in range(num_spaces):
        if len(solutions) >= num_solutions or attempts >= max_attempts:
            break

        remaining_total = num_solutions - len(solutions)
        goal_in_space = min(solutions_per_space_target, remaining_total)
        if goal_in_space <= 0:
            break

        yaw = stratified_yaw(space_idx, num_spaces, float(yaw_range_rad), rng)
        q_yaw = quat_from_yaw(yaw)
        q_target = quat_normalize(quat_multiply(q_nominal, q_yaw))
        pose_target = build_pose(target_point, q_target)

        space_attempts_start = attempts
        added_in_space = 0

        # 1) find a base IK solution for this yaw space
        base_state: Optional[RobotState] = None
        base_q: Optional[np.ndarray] = None

        while (attempts < max_attempts) and ((attempts - space_attempts_start) < space_attempt_budget):
            attempts += 1

            state = RobotState(robot_model)
            state.set_to_default_values()
            q_seed = rng.uniform_vec(lows, highs)
            state.set_joint_group_positions(group, q_seed)

            ok = state.set_from_ik(group, pose_target, tip_link, float(ik_timeout_s))
            if not ok:
                continue

            successes += 1
            state.update()
            base_state = state
            base_q = np.array(state.get_joint_group_positions(group), dtype=float)
            break

        if base_state is None or base_q is None:
            continue

        # 2) add base solution if unique
        key0 = hash_joint_vector(base_q, float(uniq_resolution_rad))
        if key0 not in uniq_keys:
            uniq_keys.add(key0)
            sol_dict = {
                "index": int(len(solutions)),
                "attempt": int(attempts),
                "target_point": {"x": float(target_point.x), "y": float(target_point.y), "z": float(target_point.z)},
                "tip_link": str(tip_link),
                "sampled_yaw_rad": float(yaw),
                "joint_names": joint_names,
                "joint_positions": [float(v) for v in base_q.tolist()],
            }
            solutions.append(sol_dict)
            solutions_obj.append(
                IKSolution(
                    index=sol_dict["index"],
                    attempt=sol_dict["attempt"],
                    sampled_yaw_rad=sol_dict["sampled_yaw_rad"],
                    joint_names=joint_names,
                    joint_positions=sol_dict["joint_positions"],
                )
            )
            added_in_space += 1

        if added_in_space >= goal_in_space:
            continue

        # 3) explore nullspace
        nvec = compute_nullspace_direction(base_state, group, tip_link, rng)

        # If nullspace unavailable, fallback to random seeds in this yaw
        if nvec is None or (int(nvec.shape[0]) != int(base_q.shape[0])):
            while (added_in_space < goal_in_space and
                   attempts < max_attempts and
                   (attempts - space_attempts_start) < space_attempt_budget):
                attempts += 1
                state = RobotState(robot_model)
                state.set_to_default_values()
                q_seed = rng.uniform_vec(lows, highs)
                state.set_joint_group_positions(group, q_seed)
                ok = state.set_from_ik(group, pose_target, tip_link, float(ik_timeout_s))
                if not ok:
                    continue
                successes += 1
                state.update()
                q = np.array(state.get_joint_group_positions(group), dtype=float)
                key = hash_joint_vector(q, float(uniq_resolution_rad))
                if key in uniq_keys:
                    continue
                uniq_keys.add(key)

                sol_dict = {
                    "index": int(len(solutions)),
                    "attempt": int(attempts),
                    "target_point": {"x": float(target_point.x), "y": float(target_point.y), "z": float(target_point.z)},
                    "tip_link": str(tip_link),
                    "sampled_yaw_rad": float(yaw),
                    "joint_names": joint_names,
                    "joint_positions": [float(v) for v in q.tolist()],
                }
                solutions.append(sol_dict)
                solutions_obj.append(
                    IKSolution(
                        index=sol_dict["index"],
                        attempt=sol_dict["attempt"],
                        sampled_yaw_rad=sol_dict["sampled_yaw_rad"],
                        joint_names=joint_names,
                        joint_positions=sol_dict["joint_positions"],
                    )
                )
                added_in_space += 1
            continue

        # Discretize around base along the nullspace direction
        half_span = nullspace_step * float(max(goal_in_space - 1, 1)) / 2.0
        num_grid = max(goal_in_space * 3, 9)

        alphas = np.linspace(-half_span, +half_span, num_grid).tolist()
        alphas += [((rng.uniform01() * 2.0) - 1.0) * half_span for _ in range(goal_in_space * 3)]
        rng.shuffle_inplace(alphas)

        for alpha in alphas:
            if added_in_space >= goal_in_space:
                break
            if attempts >= max_attempts:
                break
            if (attempts - space_attempts_start) >= space_attempt_budget:
                break
            if abs(float(alpha)) < 1e-12:
                continue

            alpha_j = float(alpha) + ((rng.uniform01() * 2.0) - 1.0) * float(nullspace_jitter)
            seed_q = base_q + alpha_j * nvec
            seed_q = clamp_to_group_bounds(seed_q, jmg)

            attempts += 1
            state = RobotState(robot_model)
            state.set_to_default_values()
            state.set_joint_group_positions(group, seed_q)

            ok = state.set_from_ik(group, pose_target, tip_link, float(ik_timeout_s))
            if not ok:
                continue

            successes += 1
            state.update()
            q = np.array(state.get_joint_group_positions(group), dtype=float)
            key = hash_joint_vector(q, float(uniq_resolution_rad))
            if key in uniq_keys:
                continue
            uniq_keys.add(key)

            sol_dict = {
                "index": int(len(solutions)),
                "attempt": int(attempts),
                "target_point": {"x": float(target_point.x), "y": float(target_point.y), "z": float(target_point.z)},
                "tip_link": str(tip_link),
                "sampled_yaw_rad": float(yaw),
                "joint_names": joint_names,
                "joint_positions": [float(v) for v in q.tolist()],
            }
            solutions.append(sol_dict)
            solutions_obj.append(
                IKSolution(
                    index=sol_dict["index"],
                    attempt=sol_dict["attempt"],
                    sampled_yaw_rad=sol_dict["sampled_yaw_rad"],
                    joint_names=joint_names,
                    joint_positions=sol_dict["joint_positions"],
                )
            )
            added_in_space += 1

    dt = time.time() - t0

    meta = {
        "group": str(group),
        "tip_link": str(tip_link),
        "named_start_for_seeding": str(named_start_for_seeding),
        "target_point": {"x": float(target_point.x), "y": float(target_point.y), "z": float(target_point.z)},
        "requested": int(num_solutions),
        "found": int(len(solutions)),
        "attempts": int(attempts),
        "ik_successes": int(successes),
        "uniq_resolution_rad": float(uniq_resolution_rad),
        "yaw_range_rad": float(yaw_range_rad),
        "ik_timeout_s": float(ik_timeout_s),
        "seed": int(seed),
        "rng": {
            "engine": "numpy.default_rng",
            "deterministic_joint_seeds": True,
            "note": "Avoid MoveIt internal RNG (RobotState.set_to_random_positions).",
        },
        "num_spaces": int(num_spaces),
        "nullspace_step": float(nullspace_step),
        "nullspace_jitter": float(nullspace_jitter),
        "elapsed_s": float(dt),
    }

    return {"meta": meta, "solutions": solutions, "solutions_obj": solutions_obj}


def save_ik_json(path: str, payload: Dict) -> None:
    """
    Save JSON in a stable format (UTF-8, readable).
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"meta": payload["meta"], "solutions": payload["solutions"]},
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
