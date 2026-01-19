from __future__ import annotations

import argparse
import math
from typing import Dict, List

import numpy as np

from ..ik.sampler_space import sample_ik_solutions, save_ik_json
from ..ik.robust_sampler import sample_ik_solutions_multi_pass
from ..planning.search import global_optimal_path_dp, greedy_path
from ..planning.time_metric import SegmentTimeModel, TotgSettings
from ..types import IKSolution, TargetPoint
from ..utils.reporting import write_robot_info_txt, write_summary_json, write_targets_json
from ..utils.robot import (
    load_robot_context,
    make_robot_state_from_joints,
    make_robot_state_from_named,
    parse_joint_positions,
)
from ..utils.targets import WorkspaceBounds, sample_one_reachable_point_fk
from ..utils.timestamp import prepare_data_paths


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ik_benchmark",
        description="MoveIt2 + Panda: sample IK solutions for random reachable points, then compare greedy vs global optimum.",
    )
    # MoveIt / robot
    p.add_argument("--group", type=str, default="panda_arm", help="Planning group name (default: panda_arm).")
    p.add_argument("--tip-link", type=str, default="", help="Tip link; empty -> infer from group.")
    p.add_argument("--named-start", type=str, default="ready", help="Named start state in SRDF (default: ready).")
    p.add_argument("--p0", type=str, default="", help="Start joint positions, e.g. '0,-0.7,0,-2.3,0,1.6,0.8'. Empty -> named-start.")

    # experiment setting
    p.add_argument("--num-points", type=int, default=3, help="Number of target points n in [2,9].")
    p.add_argument("--seed", type=int, default=7, help="Random seed for target sampling & IK sampling.")

    # IK sampling (per point)
    p.add_argument("--num-solutions", type=int, default=200, help="IK solutions per target point (default: 200).")
    p.add_argument("--num-spaces", type=int, default=20, help="Yaw spaces (default: 20).")
    p.add_argument("--max-attempts", type=int, default=20000, help="Max IK attempts per point (default: 20000).")
    p.add_argument("--ik-timeout", type=float, default=0.05, help="IK timeout per attempt (s).")
    p.add_argument("--yaw-range", type=float, default=2.0 * math.pi, help="Yaw perturbation range (rad).")
    p.add_argument("--nullspace-step", type=float, default=0.20, help="Nullspace step (rad).")
    p.add_argument("--nullspace-jitter", type=float, default=0.02, help="Nullspace jitter (rad).")
    p.add_argument("--uniq-resolution", type=float, default=1e-3, help="Uniq quantization (rad).")

    # Segment time model (stop at each waypoint)
    p.add_argument(
        "--time-model",
        type=str,
        default="auto",
        choices=["auto", "totg", "trapezoid"],
        help="Segment time model: auto=prefer MoveIt TOTG, fallback to trapezoid; "
        "totg=force MoveIt TOTG; trapezoid=analytic rest-to-rest model.",
    )
    # TOTG parameters (used when time-model is auto/totg)
    p.add_argument("--totg-vel-scale", type=float, default=1.0, help="TOTG velocity scaling factor.")
    p.add_argument("--totg-acc-scale", type=float, default=1.0, help="TOTG acceleration scaling factor.")
    p.add_argument("--totg-path-tolerance", type=float, default=0.1, help="TOTG path tolerance.")
    p.add_argument("--totg-resample-dt", type=float, default=0.1, help="TOTG resample dt (s).")
    p.add_argument("--totg-min-angle-change", type=float, default=0.001, help="TOTG min angle change (rad).")

    # Robust sampling / auto-resample (to avoid 0-solution points)
    p.add_argument(
        "--resample-max",
        type=int,
        default=200,
        help="Max resampling trials per point when IK is infeasible / insufficient.",
    )
    p.add_argument(
        "--topup-passes",
        type=int,
        default=3,
        help="How many IK sampling passes to merge (different seeds) for one point.",
    )
    p.add_argument(
        "--precheck-attempts",
        type=int,
        default=1200,
        help="Quick feasibility check budget (attempts) before doing full 200-solution sampling.",
    )
    p.add_argument(
        "--precheck-num-spaces",
        type=int,
        default=8,
        help="Quick feasibility check yaw spaces (smaller is faster).",
    )

    # output
    p.add_argument("--data-root", type=str, default="data", help="Data root directory (default: ./data).")

    # Workspace filter for random points (optional but helpful)
    p.add_argument("--ws-x", type=float, nargs=2, default=[0.15, 0.75], metavar=("X_MIN", "X_MAX"))
    p.add_argument("--ws-y", type=float, nargs=2, default=[-0.55, 0.55], metavar=("Y_MIN", "Y_MAX"))
    p.add_argument("--ws-z", type=float, nargs=2, default=[0.05, 0.85], metavar=("Z_MIN", "Z_MAX"))
    p.add_argument("--min-sep", type=float, default=0.06, help="Min separation between target points (m).")

    # Important: ignore ROS 2 launch args like --ros-args/--params-file
    args, _unknown = p.parse_known_args()
    return args


def main() -> None:
    args = _parse_args()

    n = int(args.num_points)
    if n < 2:
        n = 2
    if n > 20:
        n = 20

    data_paths = prepare_data_paths(args.data_root)
    print(f"[run] timestamp = {data_paths.timestamp}")
    print(f"[run] data dir   = {data_paths.run_dir}")

    ctx = None
    try:
        # 1) Load robot context
        ctx = load_robot_context(
            node_name="panda_ik_benchmark",
            group=str(args.group),
            tip_link=str(args.tip_link),
        )

        # 2) Build start state (p0)
        p0_list = parse_joint_positions(str(args.p0), ctx.dof)
        if p0_list is None:
            start_state = make_robot_state_from_named(ctx, str(args.named_start))
            start_label = str(args.named_start)
        else:
            start_state = make_robot_state_from_joints(ctx, p0_list)
            start_label = "custom"

        start_q = list(map(float, start_state.get_joint_group_positions(ctx.group)))

        # 3) Nominal tip orientation from start state (used as base quaternion for sampling yaw)
        tip_pose = start_state.get_pose(ctx.tip_link)
        q_nominal = (
            float(tip_pose.orientation.x),
            float(tip_pose.orientation.y),
            float(tip_pose.orientation.z),
            float(tip_pose.orientation.w),
        )

        # 4) Workspace bounds (used for candidate FK sampling)
        ws = WorkspaceBounds(
            x_min=float(args.ws_x[0]), x_max=float(args.ws_x[1]),
            y_min=float(args.ws_y[0]), y_max=float(args.ws_y[1]),
            z_min=float(args.ws_z[0]), z_max=float(args.ws_z[1]),
        )

        # Save robot info early (independent of sampled points)
        write_robot_info_txt(
            data_paths.robot_info_txt,
            group=ctx.group,
            tip_link=ctx.tip_link,
            joint_names=ctx.joint_names,
            joint_limits=[(jl.min_position, jl.max_position, jl.max_velocity, jl.max_acceleration) for jl in ctx.joint_limits],
        )

        requested = int(args.num_solutions)
        resample_max = int(args.resample_max)
        topup_passes = int(args.topup_passes)
        precheck_attempts = int(args.precheck_attempts)
        precheck_spaces = int(args.precheck_num_spaces)

        # 5) Sequentially sample p1..pn, but *only accept* points that are IK-solvable
        # under the 'ready' nominal orientation (+ yaw perturbation).
        targets: List[TargetPoint] = []
        solutions_by_point: List[List[IKSolution]] = []
        ik_meta_by_point: List[Dict] = []

        rng_points = np.random.default_rng(int(args.seed))

        for i in range(1, n + 1):
            print(f"\n[select] searching a solvable target for p{i} ...")

            best_payload = None
            best_target = None
            best_found = -1
            trials_used = 0

            for trial in range(1, resample_max + 1):
                trials_used = trial

                # 5.1) sample one FK-reachable point (position only), enforce separation
                tp = sample_one_reachable_point_fk(
                    ctx,
                    rng=rng_points,
                    existing_points=targets,
                    min_separation_m=float(args.min_sep),
                    workspace=ws,
                    max_attempts=2000,
                )

                # 5.2) quick feasibility check (fast reject for orientation-infeasible points)
                pre_payload = sample_ik_solutions(
                    ctx,
                    target_point=tp,
                    nominal_tip_quat_xyzw=q_nominal,
                    named_start_for_seeding=str(args.named_start),
                    num_solutions=50,
                    num_spaces=max(1, min(int(args.num_spaces), int(precheck_spaces))),
                    max_attempts=max(200, int(precheck_attempts)),
                    ik_timeout_s=float(args.ik_timeout),
                    yaw_range_rad=float(args.yaw_range),
                    nullspace_step=float(args.nullspace_step),
                    nullspace_jitter=float(args.nullspace_jitter),
                    uniq_resolution_rad=float(args.uniq_resolution),
                    seed=int(args.seed) + 50_000 * i + trial,
                )
                pre_found = int(pre_payload.get("meta", {}).get("found", 0))
                if pre_found <= 0:
                    # Not solvable under current orientation constraints
                    continue

                # 5.3) full sampling with multi-pass merge (try to reach requested=200)
                payload = sample_ik_solutions_multi_pass(
                    ctx,
                    target_point=tp,
                    nominal_tip_quat_xyzw=q_nominal,
                    named_start_for_seeding=str(args.named_start),
                    requested=requested,
                    passes=topup_passes,
                    pass_seed_stride=100_000,
                    num_spaces=int(args.num_spaces),
                    max_attempts=int(args.max_attempts),
                    ik_timeout_s=float(args.ik_timeout),
                    yaw_range_rad=float(args.yaw_range),
                    nullspace_step=float(args.nullspace_step),
                    nullspace_jitter=float(args.nullspace_jitter),
                    uniq_resolution_rad=float(args.uniq_resolution),
                    seed=int(args.seed) + 200_000 * i + 1000 * trial,
                )

                found = int(payload.get("meta", {}).get("found", 0))

                # Attach selection meta (useful for debugging + summary)
                payload["meta"].update(
                    {
                        "resample_trial_final": int(trial),
                        "resample_trials_used": int(trials_used),
                        "resample_max": int(resample_max),
                        "precheck_found": int(pre_found),
                        "precheck_attempts": int(pre_payload.get("meta", {}).get("attempts", 0)),
                        "precheck_ik_successes": int(pre_payload.get("meta", {}).get("ik_successes", 0)),
                        "precheck_max_attempts": int(max(200, int(precheck_attempts))),
                        "precheck_num_spaces": int(max(1, min(int(args.num_spaces), int(precheck_spaces)))),
                    }
                )

                if found > best_found:
                    best_found = found
                    best_payload = payload
                    best_target = tp

                if found >= requested:
                    # Great, we can stop resampling for this point.
                    break

            if best_payload is None or best_target is None or best_found <= 0:
                raise RuntimeError(
                    f"Failed to find any IK-solvable point for p{i} after {resample_max} trials. "
                    f"Consider relaxing workspace bounds, increasing --ik-timeout/--max-attempts, "
                    f"or reducing --num-points."
                )

            # Accept the best candidate (maybe with <200 solutions)
            if best_found < requested:
                best_payload["meta"].update(
                    {
                        "accepted_with_shortfall": True,
                        "shortfall": int(requested - best_found),
                    }
                )
            else:
                best_payload["meta"].update({"accepted_with_shortfall": False, "shortfall": 0})

            targets.append(best_target)
            solutions_by_point.append(list(best_payload["solutions_obj"]))
            ik_meta_by_point.append(dict(best_payload["meta"]))

            out_path = data_paths.ik_json_for_point(i)
            save_ik_json(str(out_path), best_payload)
            print(
                f"[select] p{i}: ({best_target.x:.3f}, {best_target.y:.3f}, {best_target.z:.3f}) "
                f"found {best_found}/{requested} (trials_used={best_payload['meta'].get('resample_trials_used')}) -> {out_path.name}"
            )

        # Save run metadata (targets only contain coordinates, per requirement)
        write_targets_json(
            data_paths.targets_json,
            timestamp=data_paths.timestamp,
            group=ctx.group,
            tip_link=ctx.tip_link,
            start_label=start_label,
            start_joint_positions=start_q,
            targets=targets,
        )

        def _make_time_model() -> SegmentTimeModel:
            return SegmentTimeModel(
                model=str(args.time_model),
                max_vel_rad_s=ctx.velocity_limits,
                max_acc_rad_s2=ctx.acceleration_limits,
                moveit_py=ctx.moveit_py,
                robot_model=ctx.robot_model,
                group=ctx.group,
                joint_names=ctx.joint_names,
                totg_settings=TotgSettings(
                    vel_scale=float(args.totg_vel_scale),
                    acc_scale=float(args.totg_acc_scale),
                    path_tolerance=float(args.totg_path_tolerance),
                    resample_dt=float(args.totg_resample_dt),
                    min_angle_change=float(args.totg_min_angle_change),
                ),
            )


        # 6) Traditional greedy
        print("\n[search] greedy baseline ...")
        greedy_res = greedy_path(
            start_q=start_q,
            solutions_by_point=solutions_by_point,
            requested_per_point=requested,
            time_model=_make_time_model(),
        )
        write_summary_json(
            data_paths.summary_traditional_json,
            title="Greedy baseline (traditional, segment-wise optimal)",
            targets=targets,
            start_label=start_label,
            start_q=start_q,
            result=greedy_res,
            ik_metas=ik_meta_by_point,
            timestamp=data_paths.timestamp,
        )
        print(f"[search] greedy summary -> {data_paths.summary_traditional_json.name}")

        # 7) Global optimum (DP, enumeration-equivalent)
        print("\n[search] global optimal (DP, enumeration-equivalent) ...")
        dp_res = global_optimal_path_dp(
            start_q=start_q,
            solutions_by_point=solutions_by_point,
            requested_per_point=requested,
            time_model=_make_time_model(),
        )
        write_summary_json(
            data_paths.summary_enumeration_json,
            title="Global optimum (DP; equivalent to exhaustive enumeration of all IK combinations)",
            targets=targets,
            start_label=start_label,
            start_q=start_q,
            result=dp_res,
            ik_metas=ik_meta_by_point,
            timestamp=data_paths.timestamp,
        )
        print(f"[search] global summary -> {data_paths.summary_enumeration_json.name}")

        print("\n[done] bye.")
    finally:
        # Always shut down MoveItPy to avoid class_loader warnings on abrupt exit.
        if ctx is not None:
            try:
                ctx.moveit_py.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
