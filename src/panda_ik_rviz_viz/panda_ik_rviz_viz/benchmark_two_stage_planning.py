#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import json
import time
import sys
import heapq
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import rclpy
from rclpy.logging import get_logger
from rclpy.utilities import remove_ros_args, try_shutdown
from rclpy.executors import ExternalShutdownException

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError

from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from moveit_configs_utils import MoveItConfigsBuilder


def rad2deg(x: float) -> float:
    return x * 180.0 / math.pi


def load_ik_json(json_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("meta", {})
    solutions = data.get("solutions", [])
    return meta, solutions


def pairs_to_joint_map(joint_names: List[str], joint_positions: List[float]) -> Dict[str, float]:
    return {str(n): float(p) for n, p in zip(joint_names, joint_positions)}


def merge_joint_positions(base: Dict[str, float], override: Dict[str, float]) -> Dict[str, float]:
    out = dict(base)
    out.update(override)
    return out


def max_joint_delta_deg(start: Dict[str, float], goal: Dict[str, float]) -> Tuple[str, float]:
    best_joint = ""
    best_delta = -1.0
    for j, g in goal.items():
        if j not in start:
            continue
        d = abs(float(g) - float(start[j]))
        if d > best_delta:
            best_delta = d
            best_joint = j
    return best_joint, rad2deg(best_delta if best_delta >= 0.0 else 0.0)


def apply_joint_map_to_state(state: RobotState, joint_map: Dict[str, float]) -> None:
    """
    Apply joint_map to RobotState with API robustness across MoveIt versions.
    """
    if hasattr(state, "joint_positions"):
        try:
            state.joint_positions = merge_joint_positions(state.joint_positions, joint_map)
            return
        except Exception:
            pass

    if hasattr(state, "set_variable_positions"):
        try:
            state.set_variable_positions(joint_map)
            return
        except Exception:
            pass

    if hasattr(state, "set_variable_position"):
        for name, value in joint_map.items():
            state.set_variable_position(str(name), float(value))
        return

    raise AttributeError("RobotState does not support setting joint positions in this environment.")


def set_start_state_compat(planning_component, *, configuration_name: Optional[str], robot_state: Optional[RobotState]) -> None:
    if hasattr(planning_component, "set_start_state"):
        if robot_state is not None:
            planning_component.set_start_state(robot_state=robot_state)
        else:
            planning_component.set_start_state(configuration_name=configuration_name)
        return

    if hasattr(planning_component, "setStartState"):
        if robot_state is not None:
            planning_component.setStartState(robot_state)
        else:
            planning_component.setStartState(configuration_name)
        return

    raise AttributeError("PlanningComponent does not provide set_start_state / setStartState.")


def set_goal_state_compat(planning_component, *, robot_state: RobotState) -> None:
    if hasattr(planning_component, "set_goal_state"):
        planning_component.set_goal_state(robot_state=robot_state)
        return

    if hasattr(planning_component, "setGoal"):
        planning_component.setGoal(robot_state)
        return

    raise AttributeError("PlanningComponent does not provide set_goal_state / setGoal.")


def execute_compat(moveit: MoveItPy, robot_trajectory, *, blocking: bool) -> None:
    if not hasattr(moveit, "execute"):
        raise AttributeError("MoveItPy has no execute() in this environment.")

    try:
        moveit.execute(robot_trajectory, blocking=blocking, controllers=[])
        return
    except TypeError:
        pass

    try:
        moveit.execute(robot_trajectory, controllers=[])
        return
    except TypeError:
        pass

    moveit.execute(robot_trajectory)


def plan_succeeded(plan_result: Any) -> bool:
    if plan_result is None:
        return False

    if hasattr(plan_result, "success"):
        try:
            return bool(plan_result.success)
        except Exception:
            pass

    if hasattr(plan_result, "error_code"):
        ec = getattr(plan_result, "error_code")
        if hasattr(ec, "val"):
            try:
                return int(ec.val) == 1
            except Exception:
                pass
        try:
            return int(ec) == 1
        except Exception:
            pass

    try:
        return bool(plan_result)
    except Exception:
        return True


def get_trajectory_duration_s(robot_traj: Any) -> float:
    if robot_traj is None:
        return float("nan")

    if hasattr(robot_traj, "duration"):
        try:
            d = robot_traj.duration
            if isinstance(d, (int, float)):
                return float(d)
            if hasattr(d, "sec") and hasattr(d, "nanosec"):
                return float(d.sec) + float(d.nanosec) * 1e-9
        except Exception:
            pass

    if hasattr(robot_traj, "get_duration"):
        try:
            d = robot_traj.get_duration()
            if isinstance(d, (int, float)):
                return float(d)
        except Exception:
            pass

    return float("nan")


def retime_trajectory(robot_traj: Any, time_param: str, vel: float, acc: float, logger) -> None:
    if robot_traj is None:
        return

    tp = (time_param or "totg").strip().lower()

    if tp == "none":
        return

    if tp == "totg":
        if hasattr(robot_traj, "apply_totg_time_parameterization"):
            robot_traj.apply_totg_time_parameterization(
                velocity_scaling_factor=float(vel),
                acceleration_scaling_factor=float(acc),
            )
        else:
            logger.warn("RobotTrajectory has no apply_totg_time_parameterization(); skip retime.")
        return

    if tp == "ruckig":
        if hasattr(robot_traj, "apply_ruckig_smoothing"):
            robot_traj.apply_ruckig_smoothing(
                velocity_scaling_factor=float(vel),
                acceleration_scaling_factor=float(acc),
            )
        else:
            logger.warn("RobotTrajectory has no apply_ruckig_smoothing(); skip retime.")
        return

    logger.warn(f"Unknown --time_param '{time_param}', treat as 'none'.")


def _resolve_summary_path(user_path: str) -> Optional[str]:
    """
    Resolve user provided path.
    - If user_path is a directory (or ends with / or \\), create a timestamped txt file inside it.
    - Otherwise, treat user_path as the output file path.
    - If output file path has no suffix, append '.txt'.
    """
    if not user_path:
        return None
    raw = str(user_path).strip()
    if not raw:
        return None

    p = Path(os.path.expanduser(raw))

    # If user clearly indicates a directory intent
    if raw.endswith(("/", "\\")) or (p.exists() and p.is_dir()):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = p / f"ik_bench_summary_{ts}.txt"
    else:
        if p.suffix == "":
            p = p.with_suffix(".txt")

    try:
        return str(p.expanduser().resolve())
    except Exception:
        return str(p.expanduser().absolute())


def write_summary_txt(user_path: str, lines: List[str], logger) -> Optional[str]:
    """
    Write summary lines to a txt file (UTF-8).
    Creates parent directories and uses atomic replace to avoid partial files.
    """
    out_path = _resolve_summary_path(user_path)
    if out_path is None:
        return None

    try:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            for ln in lines:
                f.write(str(ln).rstrip("\n") + "\n")

        os.replace(tmp_path, out_path)
        logger.info(f"Summary saved to txt: {out_path}")
        return out_path
    except Exception as e:
        logger.error(f"Failed to write summary txt '{out_path}': {e}")
        return None


def ensure_moveitcpp_pipeline_schema(cfg: Dict[str, Any], default_pipeline: str = "ompl") -> None:
    """
    Make sure config_dict follows MoveItCpp schema:
      planning_pipelines:
        pipeline_names: [...]
    """
    pp = cfg.get("planning_pipelines", None)

    if isinstance(pp, dict) and isinstance(pp.get("pipeline_names"), list) and len(pp["pipeline_names"]) > 0:
        pipeline_names = [str(x) for x in pp["pipeline_names"]]
        prp = cfg.get("plan_request_params")
        if not isinstance(prp, dict):
            cfg["plan_request_params"] = {
                "planning_attempts": 1,
                "planning_pipeline": pipeline_names[0],
                "max_velocity_scaling_factor": 1.0,
                "max_acceleration_scaling_factor": 1.0,
            }
        else:
            prp.setdefault("planning_attempts", 1)
            prp.setdefault("planning_pipeline", pipeline_names[0])
            prp.setdefault("max_velocity_scaling_factor", 1.0)
            prp.setdefault("max_acceleration_scaling_factor", 1.0)
        return

    pipeline_names: List[str] = []

    if isinstance(pp, list):
        pipeline_names = [str(x) for x in pp]
    elif isinstance(pp, dict):
        if isinstance(pp.get("planning_pipelines"), list):
            pipeline_names = [str(x) for x in pp["planning_pipelines"]]

    if not pipeline_names:
        dp = cfg.get("default_planning_pipeline", None)
        if isinstance(dp, str) and dp.strip():
            pipeline_names = [dp.strip()]

    if not pipeline_names:
        pipeline_names = [default_pipeline]

    cfg["planning_pipelines"] = {
        "pipeline_names": pipeline_names,
        "namespace": "",
    }

    prp = cfg.get("plan_request_params")
    if not isinstance(prp, dict):
        cfg["plan_request_params"] = {
            "planning_attempts": 1,
            "planning_pipeline": pipeline_names[0],
            "max_velocity_scaling_factor": 1.0,
            "max_acceleration_scaling_factor": 1.0,
        }
    else:
        prp.setdefault("planning_attempts", 1)
        prp.setdefault("planning_pipeline", pipeline_names[0])
        prp.setdefault("max_velocity_scaling_factor", 1.0)
        prp.setdefault("max_acceleration_scaling_factor", 1.0)


def build_moveit_config_dict(
    *,
    robot_name: str,
    moveit_config_pkg: str,
    default_pipeline: str,
    logger,
) -> Dict[str, Any]:
    """
    Load MoveIt configs and force use panda_ik_sampler/config/moveit_cpp_offline.yaml.
    """
    try:
        moveit_cpp_yaml = os.path.join(
            get_package_share_directory("panda_ik_sampler"),
            "config",
            "moveit_cpp_offline.yaml",
        )
    except PackageNotFoundError as e:
        raise RuntimeError(f"Package 'panda_ik_sampler' not found: {e}")

    if not os.path.exists(moveit_cpp_yaml):
        raise RuntimeError(f"MoveItCpp yaml file not found: {moveit_cpp_yaml}")

    logger.info(f"Using MoveItCpp yaml: {moveit_cpp_yaml}")

    builder = (
        MoveItConfigsBuilder(robot_name, package_name=moveit_config_pkg)
        .robot_description()
        .robot_description_semantic()
        .robot_description_kinematics()
        .planning_pipelines()
        .trajectory_execution()
        .planning_scene_monitor()
        .joint_limits()
        .moveit_cpp(file_path=moveit_cpp_yaml)
    )

    cfg = builder.to_dict()
    ensure_moveitcpp_pipeline_schema(cfg, default_pipeline=default_pipeline)
    return cfg


@dataclass
class SegmentResult:
    success: bool
    planning_time_s: float
    wall_plan_s: float
    traj_duration_s: float
    max_delta_joint: str
    max_delta_deg: float


def make_state_from_joint_map(
    robot_model,
    group: str,
    named_seed: str,
    joint_map: Dict[str, float],
) -> RobotState:
    st = RobotState(robot_model)
    st.set_to_default_values(group, named_seed)
    apply_joint_map_to_state(st, joint_map)
    st.update()
    return st


def plan_segment(
    planning_component,
    *,
    start_state: RobotState,
    start_map_for_delta: Dict[str, float],
    goal_state: RobotState,
    goal_map_for_delta: Dict[str, float],
    time_param: str,
    vel: float,
    acc: float,
    logger,
) -> Tuple[SegmentResult, Any]:
    set_start_state_compat(planning_component, configuration_name=None, robot_state=start_state)
    set_goal_state_compat(planning_component, robot_state=goal_state)

    t0 = time.perf_counter()
    plan_result = planning_component.plan()
    wall_plan = time.perf_counter() - t0

    mj, md = max_joint_delta_deg(start_map_for_delta, goal_map_for_delta)

    if not plan_succeeded(plan_result):
        return (
            SegmentResult(
                success=False,
                planning_time_s=float("nan"),
                wall_plan_s=wall_plan,
                traj_duration_s=float("nan"),
                max_delta_joint=mj,
                max_delta_deg=md,
            ),
            None,
        )

    planning_time = float(getattr(plan_result, "planning_time", float("nan")))
    robot_traj = getattr(plan_result, "trajectory", None)

    retime_trajectory(robot_traj, time_param, float(vel), float(acc), logger)
    traj_duration = get_trajectory_duration_s(robot_traj)

    return (
        SegmentResult(
            success=True,
            planning_time_s=planning_time,
            wall_plan_s=wall_plan,
            traj_duration_s=traj_duration,
            max_delta_joint=mj,
            max_delta_deg=md,
        ),
        robot_traj,
    )


def main() -> int:
    rclpy.init(args=sys.argv)
    logger = get_logger("two_stage_ik_bench.moveit_py")

    non_ros = remove_ros_args(sys.argv)[1:]

    args_map: Dict[str, Any] = {
        "json_path_b": "",
        "json_path_c": "",
        "m": 0,  # 0 means use all
        "n": 0,  # 0 means use all
        "group": "",
        "start_named": "",
        "start_joints": None,  # list[float] for panda_joint1..7
        "time_param": "totg",
        "vel": 1.0,
        "acc": 1.0,
        "topk": 10,
        "execute": 0,     # if 1, re-plan & execute the selected (global best) two segments
        "sleep": 0.2,
        "moveit_node_name": "moveit_py",
        "robot_name": "panda",
        "moveit_config_pkg": "moveit_resources_panda_moveit_config",
        "default_pipeline": "ompl",
        "summary_txt": "",  # output path for summary txt (file path or directory)
    }

    i = 0
    while i < len(non_ros):
        k = non_ros[i]
        if k == "--json_path_b":
            args_map["json_path_b"] = non_ros[i + 1]
            i += 2
        elif k == "--json_path_c":
            args_map["json_path_c"] = non_ros[i + 1]
            i += 2
        elif k == "--m":
            args_map["m"] = int(non_ros[i + 1])
            i += 2
        elif k == "--n":
            args_map["n"] = int(non_ros[i + 1])
            i += 2
        elif k == "--group":
            args_map["group"] = non_ros[i + 1]
            i += 2
        elif k == "--start_named":
            args_map["start_named"] = non_ros[i + 1]
            i += 2
        elif k == "--start_joints":
            vals = [float(x) for x in non_ros[i + 1 : i + 8]]
            if len(vals) != 7:
                logger.error("--start_joints expects 7 values (panda_joint1..7).")
                try_shutdown()
                return 2
            args_map["start_joints"] = vals
            i += 8
        elif k == "--time_param":
            args_map["time_param"] = non_ros[i + 1].strip().lower()
            i += 2
        elif k == "--vel":
            args_map["vel"] = float(non_ros[i + 1])
            i += 2
        elif k == "--acc":
            args_map["acc"] = float(non_ros[i + 1])
            i += 2
        elif k == "--topk":
            args_map["topk"] = int(non_ros[i + 1])
            i += 2
        elif k == "--execute":
            args_map["execute"] = int(non_ros[i + 1])
            i += 2
        elif k == "--sleep":
            args_map["sleep"] = float(non_ros[i + 1])
            i += 2
        elif k == "--moveit_node_name":
            args_map["moveit_node_name"] = non_ros[i + 1]
            i += 2
        elif k == "--robot_name":
            args_map["robot_name"] = non_ros[i + 1]
            i += 2
        elif k == "--moveit_config_pkg":
            args_map["moveit_config_pkg"] = non_ros[i + 1]
            i += 2
        elif k == "--default_pipeline":
            args_map["default_pipeline"] = non_ros[i + 1]
            i += 2
        elif k == "--summary_txt":
            args_map["summary_txt"] = non_ros[i + 1]
            i += 2
        else:
            logger.warn(f"Unknown arg '{k}', ignored.")
            i += 1

    json_path_b = args_map["json_path_b"]
    json_path_c = args_map["json_path_c"]
    if not json_path_b or not json_path_c:
        logger.error(
            "Missing --json_path_b or --json_path_c.\n"
            "Example: --json_path_b /path/B.json --json_path_c /path/C.json"
        )
        try_shutdown()
        return 2

    meta_b, sols_b = load_ik_json(json_path_b)
    meta_c, sols_c = load_ik_json(json_path_c)

    m_limit = int(args_map["m"])
    n_limit = int(args_map["n"])
    m = len(sols_b) if m_limit <= 0 else min(m_limit, len(sols_b))
    n = len(sols_c) if n_limit <= 0 else min(n_limit, len(sols_c))

    group = args_map["group"] or meta_b.get("group") or meta_c.get("group") or "panda_arm"
    start_named = args_map["start_named"] or meta_b.get("named_start") or meta_c.get("named_start") or "ready"

    logger.info(
        f"Loaded B IK JSON: path='{json_path_b}', m={m} (of {len(sols_b)}) | "
        f"Loaded C IK JSON: path='{json_path_c}', n={n} (of {len(sols_c)})"
    )
    logger.info(f"Using group='{group}', start_named='{start_named}'")

    moveit: Optional[MoveItPy] = None

    # Store durations
    ab_dur: List[float] = [float("nan")] * m
    bc_dur: List[List[float]] = [[float("nan")] * n for _ in range(m)]

    # Keep joint maps & attempts for referencing
    b_maps: List[Dict[str, float]] = []
    c_maps: List[Dict[str, float]] = []
    b_attempts: List[int] = []
    c_attempts: List[int] = []

    # RobotStates for B/C (built once)
    b_states: List[Optional[RobotState]] = [None] * m
    c_states: List[Optional[RobotState]] = [None] * n

    # For optional execution: remember best pair indices
    best_total = float("inf")
    best_pair: Tuple[int, int] = (-1, -1)

    # Keep top-k pairs by total time
    topk = max(1, int(args_map["topk"]))
    top_heap: List[Tuple[float, int, int]] = []  # (-total, i, j)

    try:
        moveit_config_dict = build_moveit_config_dict(
            robot_name=str(args_map["robot_name"]),
            moveit_config_pkg=str(args_map["moveit_config_pkg"]),
            default_pipeline=str(args_map["default_pipeline"]),
            logger=logger,
        )

        moveit = MoveItPy(
            node_name=str(args_map["moveit_node_name"]),
            config_dict=moveit_config_dict,
        )

        planning_component = moveit.get_planning_component(group)
        robot_model = moveit.get_robot_model()

        # Start state A
        start_state_a = RobotState(robot_model)
        start_state_a.set_to_default_values(group, start_named)
        start_state_a.update()

        if args_map["start_joints"] is not None:
            override = {
                "panda_joint1": args_map["start_joints"][0],
                "panda_joint2": args_map["start_joints"][1],
                "panda_joint3": args_map["start_joints"][2],
                "panda_joint4": args_map["start_joints"][3],
                "panda_joint5": args_map["start_joints"][4],
                "panda_joint6": args_map["start_joints"][5],
                "panda_joint7": args_map["start_joints"][6],
            }
            apply_joint_map_to_state(start_state_a, override)
            start_state_a.update()
            logger.info("Start state A overridden by --start_joints (rad).")

        start_pos_map = dict(getattr(start_state_a, "joint_positions", {}))
        if not start_pos_map:
            logger.warn("Start state's joint_positions not available; joint filtering may be weak.")

        # Pre-build B joint maps and states
        for bi in range(m):
            sol = sols_b[bi]
            attempt = int(sol.get("attempt", bi + 1))
            jnames = sol.get("joint_names", [])
            jpos = sol.get("joint_positions", [])
            if len(jnames) != len(jpos):
                logger.warn(f"[B{bi+1}] joint_names({len(jnames)}) != joint_positions({len(jpos)}); zip() truncates.")

            jm_raw = pairs_to_joint_map(jnames, jpos)
            jm = {k: v for k, v in jm_raw.items() if (not start_pos_map) or (k in start_pos_map)}
            b_maps.append(jm)
            b_attempts.append(attempt)

            if not jm:
                logger.error(f"[B{bi+1}] No usable joint targets after filtering; will be treated as invalid.")
                b_states[bi] = None
            else:
                b_states[bi] = make_state_from_joint_map(robot_model, group, start_named, jm)

        # Pre-build C joint maps and states
        for cj in range(n):
            sol = sols_c[cj]
            attempt = int(sol.get("attempt", cj + 1))
            jnames = sol.get("joint_names", [])
            jpos = sol.get("joint_positions", [])
            if len(jnames) != len(jpos):
                logger.warn(f"[C{cj+1}] joint_names({len(jnames)}) != joint_positions({len(jpos)}); zip() truncates.")

            jm_raw = pairs_to_joint_map(jnames, jpos)
            jm = {k: v for k, v in jm_raw.items() if (not start_pos_map) or (k in start_pos_map)}
            c_maps.append(jm)
            c_attempts.append(attempt)

            if not jm:
                logger.error(f"[C{cj+1}] No usable joint targets after filtering; will be treated as invalid.")
                c_states[cj] = None
            else:
                c_states[cj] = make_state_from_joint_map(robot_model, group, start_named, jm)

        # -------------------------
        # 1) Plan A -> B_i for all i
        # -------------------------
        logger.info("========== Stage-1: Plan A -> B_i ==========")

        for bi in range(m):
            if b_states[bi] is None:
                ab_dur[bi] = float("nan")
                continue

            # Use compat wrapper (start by RobotState A, goal by RobotState B_i)
            set_start_state_compat(planning_component, configuration_name=start_named, robot_state=start_state_a)
            set_goal_state_compat(planning_component, robot_state=b_states[bi])

            t0 = time.perf_counter()
            plan_result = planning_component.plan()
            wall_plan = time.perf_counter() - t0

            mj, md = max_joint_delta_deg(start_pos_map, b_maps[bi])

            if not plan_succeeded(plan_result):
                ab_dur[bi] = float("nan")
                logger.error(
                    f"[A->B{bi+1}] FAIL | attempt={b_attempts[bi]} | wall_plan={wall_plan:.4f}s | "
                    f"max_delta={md:.2f}deg @ {mj}"
                )
                continue

            planning_time = float(getattr(plan_result, "planning_time", float("nan")))
            robot_traj = getattr(plan_result, "trajectory", None)

            retime_trajectory(robot_traj, args_map["time_param"], float(args_map["vel"]), float(args_map["acc"]), logger)
            dur = get_trajectory_duration_s(robot_traj)
            ab_dur[bi] = dur

            logger.info(
                f"[A->B{bi+1}] OK | attempt={b_attempts[bi]} | plan_time={planning_time:.4f}s | "
                f"wall_plan={wall_plan:.4f}s | traj_duration={dur:.4f}s | max_delta={md:.2f}deg @ {mj}"
            )

        # Find greedy best B: min A->B time (t3)
        best_b_idx = -1
        t3 = float("inf")
        for bi, dur in enumerate(ab_dur):
            if not math.isnan(dur) and dur < t3:
                t3 = dur
                best_b_idx = bi

        if best_b_idx < 0:
            logger.error("No successful A->B plans. Cannot proceed to greedy selection.")
        else:
            logger.info(f"Greedy Stage-1 best: B{best_b_idx+1} (attempt={b_attempts[best_b_idx]}) with t3={t3:.4f}s")

        # -------------------------
        # 2) Plan B_i -> C_j for all (i, j)
        # -------------------------
        logger.info("========== Stage-2: Plan B_i -> C_j for all pairs ==========")

        bc_success = 0
        bc_total = 0

        for bi in range(m):
            if b_states[bi] is None:
                continue

            for cj in range(n):
                bc_total += 1

                if c_states[cj] is None:
                    bc_dur[bi][cj] = float("nan")
                    continue

                set_start_state_compat(planning_component, configuration_name=None, robot_state=b_states[bi])
                set_goal_state_compat(planning_component, robot_state=c_states[cj])

                t0 = time.perf_counter()
                plan_result = planning_component.plan()
                wall_plan = time.perf_counter() - t0

                mj, md = max_joint_delta_deg(b_maps[bi], c_maps[cj])

                if not plan_succeeded(plan_result):
                    bc_dur[bi][cj] = float("nan")
                    # Keep logs light; you can switch to info if you want full matrix logs
                    logger.warn(
                        f"[B{bi+1}->C{cj+1}] FAIL | attempts=({b_attempts[bi]},{c_attempts[cj]}) | "
                        f"wall_plan={wall_plan:.4f}s | max_delta={md:.2f}deg @ {mj}"
                    )
                    continue

                planning_time = float(getattr(plan_result, "planning_time", float("nan")))
                robot_traj = getattr(plan_result, "trajectory", None)

                retime_trajectory(robot_traj, args_map["time_param"], float(args_map["vel"]), float(args_map["acc"]), logger)
                dur = get_trajectory_duration_s(robot_traj)
                bc_dur[bi][cj] = dur
                bc_success += 1

                # Total time for pair (requires A->B success)
                if not math.isnan(ab_dur[bi]) and not math.isnan(dur):
                    total = ab_dur[bi] + dur

                    # Global best pair
                    if total < best_total:
                        best_total = total
                        best_pair = (bi, cj)

                    # Top-k
                    if len(top_heap) < topk:
                        heapq.heappush(top_heap, (-total, bi, cj))
                    else:
                        # Replace worst in heap if this total is better
                        worst_neg, _, _ = top_heap[0]
                        worst_total = -worst_neg
                        if total < worst_total:
                            heapq.heapreplace(top_heap, (-total, bi, cj))

        logger.info(f"Stage-2 done: BC success {bc_success}/{bc_total} planned pairs")

        # Greedy Stage-2 best C for chosen best_b_idx: min B_best -> C time (t4)
        best_c_idx = -1
        t4 = float("inf")
        if best_b_idx >= 0:
            for cj in range(n):
                dur = bc_dur[best_b_idx][cj]
                if not math.isnan(dur) and dur < t4:
                    t4 = dur
                    best_c_idx = cj

            if best_c_idx < 0:
                logger.error(f"No successful plans from greedy B{best_b_idx+1} to any C_j.")
            else:
                logger.info(
                    f"Greedy Stage-2 best: C{best_c_idx+1} (attempt={c_attempts[best_c_idx]}) with t4={t4:.4f}s"
                )

        # -------------------------
        # Summary (log + txt)
        # -------------------------
        summary_lines: List[str] = []

        def emit(line: str) -> None:
            logger.info(line)
            # logger.info("start writing summary")
            summary_lines.append(line)

        emit("========== Summary ==========")
        emit(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
        emit(f"json_path_b: {json_path_b}")
        emit(f"json_path_c: {json_path_c}")
        emit(f"group: {group}")
        emit(f"start_named: {start_named}")
        emit(
            f"time_param: {str(args_map['time_param'])} | "
            f"vel: {float(args_map['vel']):.3f} | acc: {float(args_map['acc']):.3f}"
        )
        emit(
            f"m: {m} (of {len(sols_b)}) | n: {n} (of {len(sols_c)}) | "
            f"topk: {topk} | execute: {int(args_map['execute'])}"
        )

        ab_ok = sum(1 for x in ab_dur if not math.isnan(x))
        emit(f"A->B success: {ab_ok}/{m}")

        if best_b_idx >= 0:
            emit(f"t3 (min A->B): {t3:.4f}s at B{best_b_idx+1} (attempt={b_attempts[best_b_idx]})")

        emit(f"B->C success: {bc_success}/{bc_total}")

        if best_b_idx >= 0 and best_c_idx >= 0:
            emit(f"t4 (min B_greedy->C): {t4:.4f}s at C{best_c_idx+1} (attempt={c_attempts[best_c_idx]})")
            emit(f"Greedy total (t3+t4): {(t3 + t4):.4f}s")

        if best_pair[0] >= 0:
            bi, cj = best_pair
            emit(
                f"Global best pair: (B{bi+1}, C{cj+1}) | "
                f"t1={ab_dur[bi]:.4f}s, t2={bc_dur[bi][cj]:.4f}s, total={best_total:.4f}s | "
                f"attempts=({b_attempts[bi]},{c_attempts[cj]})"
            )

        if top_heap:
            top_sorted = sorted([(-neg, bi, cj) for (neg, bi, cj) in top_heap], key=lambda x: x[0])
            emit(f"Top-{min(topk, len(top_sorted))} pairs by total time:")
            for rank, (tot, bi, cj) in enumerate(top_sorted, start=1):
                emit(
                    f"  #{rank}: (B{bi+1}, C{cj+1}) | "
                    f"t1={ab_dur[bi]:.4f}s, t2={bc_dur[bi][cj]:.4f}s, total={tot:.4f}s"
                )

        write_summary_txt(str(args_map.get("summary_txt", "")), summary_lines, logger)

        # Optional execute: re-plan & execute global best two segments
        if int(args_map["execute"]) == 1 and best_pair[0] >= 0:
            bi, cj = best_pair
            logger.warn(
                "Execute enabled: will re-plan and execute GLOBAL BEST pair in two segments (A->B_best, B_best->C_best)."
            )

            # Rebuild states (safe)
            b_state = b_states[bi]
            c_state = c_states[cj]
            if b_state is None or c_state is None:
                logger.error("Selected states invalid for execution. Skip.")
            else:
                # Plan A->B_best
                seg1, traj1 = plan_segment(
                    planning_component,
                    start_state=start_state_a,
                    start_map_for_delta=start_pos_map,
                    goal_state=b_state,
                    goal_map_for_delta=b_maps[bi],
                    time_param=str(args_map["time_param"]),
                    vel=float(args_map["vel"]),
                    acc=float(args_map["acc"]),
                    logger=logger,
                )
                if not seg1.success or traj1 is None:
                    logger.error("Execution: plan A->B_best failed. Abort execution.")
                else:
                    execute_compat(moveit, traj1, blocking=True)
                    time.sleep(float(args_map["sleep"]))

                    # Plan B_best->C_best
                    seg2, traj2 = plan_segment(
                        planning_component,
                        start_state=b_state,
                        start_map_for_delta=b_maps[bi],
                        goal_state=c_state,
                        goal_map_for_delta=c_maps[cj],
                        time_param=str(args_map["time_param"]),
                        vel=float(args_map["vel"]),
                        acc=float(args_map["acc"]),
                        logger=logger,
                    )
                    if not seg2.success or traj2 is None:
                        logger.error("Execution: plan B_best->C_best failed. Abort execution.")
                    else:
                        execute_compat(moveit, traj2, blocking=True)
                        time.sleep(float(args_map["sleep"]))

    except (KeyboardInterrupt, ExternalShutdownException):
        logger.warn("Interrupted, exiting.")
    except Exception as e:
        logger.error(f"Exception: {e}")
        raise
    finally:
        try:
            if moveit is not None and hasattr(moveit, "shutdown"):
                moveit.shutdown()
        except Exception as e:
            logger.warn(f"MoveItPy shutdown exception: {e}")

        try_shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())