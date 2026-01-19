#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import json
import time
import sys
from dataclasses import dataclass
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


def ensure_moveitcpp_pipeline_schema(cfg: Dict[str, Any], default_pipeline: str = "ompl") -> None:
    """
    MoveItPy/MoveItCpp 需要：
      planning_pipelines:
        pipeline_names: [...]
    如果已有 pipeline_names，就不覆盖；否则从旧结构推断并补齐。
    相关配置方式见 MoveIt 官方 Motion Planning Python API 教程。:contentReference[oaicite:2]{index=2}
    """
    pp = cfg.get("planning_pipelines", None)

    # 如果已经是 MoveItCpp 想要的结构，就只补 plan_request_params（如缺）
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

    # 否则：从旧结构/默认值推断
    pipeline_names: List[str] = []

    if isinstance(pp, list):
        pipeline_names = [str(x) for x in pp]
    elif isinstance(pp, dict):
        # MoveItConfigsBuilder.planning_pipelines() 的旧结构是：
        # planning_pipelines: { planning_pipelines: [...], default_planning_pipeline: ... , ompl: {...}, ... }
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
    用 moveit_config_pkg 加载 URDF/SRDF/kinematics/ompl 等，
    并强制使用 panda_ik_sampler/config/moveit_cpp_offline.yaml 作为 moveit_cpp 参数文件。:contentReference[oaicite:3]{index=3}
    """
    # 你指定的 moveit_cpp_offline.yaml（强制使用）
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
        # 关键：用你给的 moveit_cpp_offline.yaml 替代默认 config/moveit_cpp.yaml
        .moveit_cpp(file_path=moveit_cpp_yaml)
    )

    cfg = builder.to_dict()

    # 只在缺失 schema 时补齐，避免覆盖你 moveit_cpp_offline.yaml 里已经写好的 planning_pipelines
    ensure_moveitcpp_pipeline_schema(cfg, default_pipeline=default_pipeline)

    return cfg


@dataclass
class BenchResult:
    idx: int
    attempt: int
    success: bool
    planning_time_s: float
    wall_plan_s: float
    traj_duration_s: float
    wall_exec_s: float
    max_delta_joint: str
    max_delta_deg: float


def main() -> int:
    rclpy.init(args=sys.argv)
    logger = get_logger("ik_bench.moveit_py")

    non_ros = remove_ros_args(sys.argv)[1:]

    # Manual args parsing (kept minimal)
    args_map: Dict[str, Any] = {
        "json_path": "",
        "n": 8,
        "group": "",
        "start_named": "",
        "start_joints": None,  # list[float]
        "time_param": "totg",
        "vel": 1.0,
        "acc": 1.0,
        "execute": 0,
        "sleep": 0.2,
        "moveit_node_name": "moveit_py",
        "robot_name": "panda",
        "moveit_config_pkg": "moveit_resources_panda_moveit_config",
        "default_pipeline": "ompl",
    }

    i = 0
    while i < len(non_ros):
        k = non_ros[i]
        if k == "--json_path":
            args_map["json_path"] = non_ros[i + 1]
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
        else:
            logger.warn(f"Unknown arg '{k}', ignored.")
            i += 1

    json_path = args_map["json_path"]
    if not json_path:
        logger.error("Missing --json_path. Example: --json_path /home/cao/panda_ik_solutions.json")
        try_shutdown()
        return 2

    meta, solutions = load_ik_json(json_path)
    n = min(int(args_map["n"]), len(solutions))

    group = args_map["group"] or meta.get("group", "panda_arm")
    start_named = args_map["start_named"] or meta.get("named_start", "ready")

    logger.info(f"Loaded IK JSON: path='{json_path}', group='{group}', start_named='{start_named}', n={n}")

    moveit: Optional[MoveItPy] = None
    results: List[BenchResult] = []

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

        # Start state
        start_state = RobotState(robot_model)
        start_state.set_to_default_values(group, start_named)
        start_state.update()

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
            apply_joint_map_to_state(start_state, override)
            start_state.update()
            logger.info("Start state overridden by --start_joints (rad).")

        start_pos_map = dict(getattr(start_state, "joint_positions", {}))

        for idx in range(n):
            sol = solutions[idx]
            attempt = int(sol.get("attempt", idx + 1))
            jnames = sol.get("joint_names", [])
            jpos = sol.get("joint_positions", [])

            if len(jnames) != len(jpos):
                logger.warn(
                    f"[{idx+1}] joint_names({len(jnames)}) != joint_positions({len(jpos)}). "
                    "Will truncate by zip()."
                )

            goal_map_raw = pairs_to_joint_map(jnames, jpos)

            # Filter only joints existing in start state (avoid panda_joint8)
            goal_map = {k: v for k, v in goal_map_raw.items() if k in start_pos_map}
            if not goal_map:
                logger.error(f"[{idx+1}] No usable joint targets after filtering. Skip.")
                continue

            goal_state = RobotState(robot_model)
            goal_state.set_to_default_values(group, start_named)
            apply_joint_map_to_state(goal_state, goal_map)
            goal_state.update()

            set_start_state_compat(planning_component, configuration_name=start_named, robot_state=start_state)
            set_goal_state_compat(planning_component, robot_state=goal_state)

            t0 = time.perf_counter()
            plan_result = planning_component.plan()
            wall_plan = time.perf_counter() - t0

            mj, md = max_joint_delta_deg(start_pos_map, goal_map)

            if not plan_succeeded(plan_result):
                results.append(
                    BenchResult(
                        idx=idx + 1,
                        attempt=attempt,
                        success=False,
                        planning_time_s=float("nan"),
                        wall_plan_s=wall_plan,
                        traj_duration_s=float("nan"),
                        wall_exec_s=float("nan"),
                        max_delta_joint=mj,
                        max_delta_deg=md,
                    )
                )
                logger.error(f"[{idx+1}] Planning failed. wall_plan={wall_plan:.4f}s")
                continue

            planning_time = float(getattr(plan_result, "planning_time", float("nan")))
            robot_traj = getattr(plan_result, "trajectory", None)

            retime_trajectory(robot_traj, args_map["time_param"], float(args_map["vel"]), float(args_map["acc"]), logger)
            traj_duration = get_trajectory_duration_s(robot_traj)

            wall_exec = float("nan")
            if int(args_map["execute"]) == 1 and robot_traj is not None:
                t1 = time.perf_counter()
                execute_compat(moveit, robot_traj, blocking=True)
                wall_exec = time.perf_counter() - t1
                time.sleep(float(args_map["sleep"]))

            results.append(
                BenchResult(
                    idx=idx + 1,
                    attempt=attempt,
                    success=True,
                    planning_time_s=planning_time,
                    wall_plan_s=wall_plan,
                    traj_duration_s=traj_duration,
                    wall_exec_s=wall_exec,
                    max_delta_joint=mj,
                    max_delta_deg=md,
                )
            )

            logger.info(
                f"[{idx+1}] OK | attempt={attempt} | planning_time={planning_time:.4f}s | "
                f"wall_plan={wall_plan:.4f}s | traj_duration={traj_duration:.4f}s | "
                f"max_delta={md:.2f}deg @ {mj}"
                + (f" | wall_exec={wall_exec:.4f}s" if int(args_map["execute"]) == 1 else "")
            )

        logger.info("========== Summary ==========")
        for r in results:
            if not r.success:
                logger.info(
                    f"[{r.idx}] FAIL | attempt={r.attempt} | wall_plan={r.wall_plan_s:.4f}s | "
                    f"max_delta={r.max_delta_deg:.2f}deg @ {r.max_delta_joint}"
                )
            else:
                extra = f" | wall_exec={r.wall_exec_s:.4f}s" if not math.isnan(r.wall_exec_s) else ""
                logger.info(
                    f"[{r.idx}] OK   | attempt={r.attempt} | plan={r.planning_time_s:.4f}s | "
                    f"wall_plan={r.wall_plan_s:.4f}s | traj={r.traj_duration_s:.4f}s | "
                    f"max_delta={r.max_delta_deg:.2f}deg @ {r.max_delta_joint}{extra}"
                )

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
