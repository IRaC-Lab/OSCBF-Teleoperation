#!/usr/bin/env python3
"""ROS2 Controller Node for the Franka

Publishes:
- Joint torques

Subscribes:
- Joint states (position, velocity)
- Desired end-effector state (position, orientation, velocity, angular velocity)
"""

import signal
import time
import sys
import csv
import queue
import threading
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
from cbfpy import CBF, CBFConfig

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from oscbf_control_msgs.msg import EEState

from oscbf_control.utils.rotations_and_transforms import xyzw_to_rotation_numpy
from oscbf.core.manipulator import Manipulator, load_panda
from oscbf.core.controllers import PoseTaskTorqueController

jax.config.update("jax_enable_x64", True)


@jax.tree_util.register_static
class DemoConfig(CBFConfig):
    """CBF Config for demoing OSCBF on the Franka hardware.

    The CBF state augments the robot state with the obstacle motion state:
        z = [q, qdot, p_obs, v_obs, a_obs]
    This lets the high-order CBF derivatives include obstacle velocity and
    acceleration through CBFpy's automatic Lie derivative calculations.
    """

    def __init__(
        self,
        robot: Manipulator,
        whole_body_pos_min: ArrayLike,
        whole_body_pos_max: ArrayLike,
        pos_obj_weight: float = 1.0,
        rot_obj_weight: float = 1.0,
        joint_obj_weight: float = 1.0,
        compensate_centrifugal_coriolis: bool = True,
    ):
        assert isinstance(robot, Manipulator)
        assert isinstance(pos_obj_weight, (tuple, float)) and pos_obj_weight >= 0
        assert isinstance(rot_obj_weight, (tuple, float)) and rot_obj_weight >= 0
        assert isinstance(joint_obj_weight, (tuple, float)) and joint_obj_weight >= 0
        assert isinstance(compensate_centrifugal_coriolis, bool)

        self.robot = robot
        self.num_joints = self.robot.num_joints
        self.task_dim = 6
        self.is_redundant = self.robot.num_joints > self.task_dim
        self.compensate_centrifugal_coriolis = compensate_centrifugal_coriolis
        self.pos_obj_weight = float(pos_obj_weight)
        self.rot_obj_weight = float(rot_obj_weight)
        self.joint_space_obj_weight = float(joint_obj_weight)
        self.W_T_W_task_diag = tuple(
            np.array([self.pos_obj_weight] * 3 + [self.rot_obj_weight] * 3) ** 2
        )
        self.W_T_W_joint_diag = tuple(
            np.array([self.joint_space_obj_weight] * self.num_joints) ** 2
        )

        self.q_min = robot.joint_lower_limits
        self.q_max = robot.joint_upper_limits
        self.singularity_tol = 1e-2
        self.whole_body_pos_min = np.asarray(whole_body_pos_min, dtype=float)
        self.whole_body_pos_max = np.asarray(whole_body_pos_max, dtype=float)

        # Default obstacle geometry and initial moving-obstacle center.
        self.collision_pos = np.array([[0.4, -0.10, 0.09]], dtype=float)
        self.collision_radii = np.array([0.2], dtype=float)

        self.collision_margin = 0.0

        super().__init__(
            n=self.num_joints * 2 + 9,
            m=self.num_joints,
            u_min=-np.asarray(robot.joint_max_forces),
            u_max=np.asarray(robot.joint_max_forces),
        )

    def f(self, z, *args, **kwargs):
        q = z[: self.num_joints]
        q_dot = z[self.num_joints : self.num_joints * 2]
        obstacle_vel = z[self.num_joints * 2 + 3 : self.num_joints * 2 + 6]
        obstacle_acc = z[self.num_joints * 2 + 6 : self.num_joints * 2 + 9]

        M = self.robot.mass_matrix(q)
        M_inv = jnp.linalg.inv(M)
        bias = self.robot.gravity_vector(q)
        if self.compensate_centrifugal_coriolis:
            bias += self.robot.centrifugal_coriolis_vector(q, q_dot)

        return jnp.concatenate(
            [
                q_dot,
                -M_inv @ bias,
                obstacle_vel,
                obstacle_acc,
                jnp.zeros(3),
            ]
        )

    def g(self, z, *args, **kwargs):
        q = z[: self.num_joints]
        M = self.robot.mass_matrix(q)
        M_inv = jnp.linalg.inv(M)
        return jnp.block(
            [
                [jnp.zeros((self.num_joints, self.m))],
                [M_inv],
                [jnp.zeros((9, self.m))],
            ]
        )

    def _P(self, z, *args, **kwargs):
        q = z[: self.num_joints]
        transforms = self.robot.joint_to_world_transforms(q)
        M = self.robot._mass_matrix(transforms)
        M_inv = jnp.linalg.inv(M)
        J = self.robot._ee_jacobian(transforms)
        task_inertia_inv = J @ M_inv @ J.T
        task_inertia = jnp.linalg.inv(task_inertia_inv)
        J_bar = M_inv @ J.T @ task_inertia
        NT = jnp.eye(self.num_joints) - J.T @ J_bar.T

        W_T_W_joint = jnp.diag(jnp.asarray(self.W_T_W_joint_diag))
        W_T_W_task = jnp.diag(jnp.asarray(self.W_T_W_task_diag))

        return (
            NT.T @ M_inv.T @ W_T_W_joint @ M_inv @ NT
            + M_inv.T @ J.T @ W_T_W_task @ J @ M_inv
        )

    def P(self, z, u_des, *args, **kwargs):
        return self._P(z)

    def q(self, z, u_des, *args, **kwargs):
        return -u_des.T @ self._P(z)

    def h_2(self, z, *args, **kwargs):
        q = z[: self.num_joints]
        collision_pos = z[self.num_joints * 2 : self.num_joints * 2 + 3][None, :]
        q_min = jnp.asarray(self.q_min)
        q_max = jnp.asarray(self.q_max)

        h_joint_limits = jnp.concatenate([q_max - q, q - q_min])

        sigmas = jax.lax.linalg.svd(self.robot.ee_jacobian(q), compute_uv=False)
        h_singularity = jnp.array([jnp.prod(sigmas) - self.singularity_tol])

        robot_collision_pos_rad = self.robot.link_collision_data(q)
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3]

        robot_num_pts = robot_collision_positions.shape[0]

        h_whole_body_upper = (
            jnp.tile(self.whole_body_pos_max, (robot_num_pts, 1))
            - robot_collision_positions
            - robot_collision_radii[:, None]
        ).ravel()

        h_whole_body_lower = (
            robot_collision_positions
            - jnp.tile(self.whole_body_pos_min, (robot_num_pts, 1))
            - robot_collision_radii[:, None]
        ).ravel()

        obs_pos = jnp.asarray(collision_pos)           # (Nobs, 3)
        obs_radii = jnp.asarray(self.collision_radii)  # (Nobs,)

        diff = robot_collision_positions[:, None, :] - obs_pos[None, :, :]
        dist_sq = jnp.sum(diff**2, axis=-1)

        safe_dist = (
            robot_collision_radii[:, None]
            + obs_radii[None, :]
            + self.collision_margin
        )
        h_obstacle = (dist_sq - safe_dist**2).ravel()

        return jnp.concatenate(
            [
                h_joint_limits,
                h_singularity,
                h_whole_body_upper,
                h_whole_body_lower,
                h_obstacle,
            ]
        )

    def h_1(self, z, *args, **kwargs):
        qdot = z[self.num_joints : self.num_joints * 2]
        joint_max_vels = jnp.asarray(self.robot.joint_max_velocities)
        qdot_max = joint_max_vels
        qdot_min = -joint_max_vels
        return jnp.concatenate([qdot_max - qdot, qdot - qdot_min])

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


@partial(jax.jit, static_argnums=(0, 1, 2))
def compute_control(
    robot: Manipulator,
    osc_controller: PoseTaskTorqueController,
    cbf: CBF,
    z_aug: ArrayLike,
    z_ee_des: ArrayLike,
):
    q = z_aug[: robot.num_joints]
    qdot = z_aug[robot.num_joints : robot.num_joints * 2]
    M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)

    nullspace_posture_goal = jnp.array(
        [
            0.0,
            -jnp.pi / 6,
            0.0,
            -3 * jnp.pi / 4,
            0.0,
            5 * jnp.pi / 9,
            0.0,
        ]
    )

    u_nom = osc_controller(
        q,
        qdot,
        pos=ee_tmat[:3, 3],
        rot=ee_tmat[:3, :3],
        des_pos=z_ee_des[:3],
        des_rot=jnp.reshape(z_ee_des[3:12], (3, 3)),
        des_vel=z_ee_des[12:15],
        des_omega=z_ee_des[15:18],
        des_accel=jnp.zeros(3),
        des_alpha=jnp.zeros(3),
        des_q=nullspace_posture_goal,
        des_qdot=jnp.zeros(robot.num_joints),
        J=J,
        M=M,
        M_inv=M_inv,
        g=g,
        c=c,
    )

    tau_filtered = cbf.safety_filter(z_aug, u_nom)

    # Franka side adds gravity compensation internally.
    return tau_filtered - g, u_nom, tau_filtered, ee_tmat[:3, 3]


class OSCBFNode(Node):
    def __init__(
        self,
        whole_body_pos_min: ArrayLike = (-0.5, -0.5, 0.0),
        whole_body_pos_max: ArrayLike = (0.75, 0.5, 1.0),
    ):
        super().__init__("oscbf_node")
        self.get_logger().info("Initializing OSCBF Node...")
        self.declare_parameter("enable_diagnostics", True)
        self.declare_parameter("diagnostics_rate_hz", 20.0)

        whole_body_pos_min = np.asarray(whole_body_pos_min)
        whole_body_pos_max = np.asarray(whole_body_pos_max)
        assert whole_body_pos_min.shape == (3,)
        assert whole_body_pos_max.shape == (3,)

        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.torque_pub = self.create_publisher(
            Float64MultiArray, "franka/torque_command", qos_profile
        )
        self.joint_state_sub = self.create_subscription(
            JointState, "franka/joint_states", self.joint_state_callback, qos_profile
        )
        self.ee_state_sub = self.create_subscription(
            EEState, "ee_state", self.ee_state_callback, qos_profile
        )
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.control_freq = 1000
        self.timer = self.create_timer(1 / self.control_freq, self.publish_control)

        self.last_joint_state = None
        self.last_ee_state = None
        self.last_tau_cmd = np.zeros(7)
        self.log_count = 0

        self.startup_hold_sec = 2.0
        self.start_time = time.time()
        self.initialized_desired_state = False
        self.safe_desired_ee_state = None

        self.max_abs_tau = np.array([100.0, 100.0, 100.0, 100.0, 80.0, 80.0, 80.0])
        self.max_delta_tau = 100

        self.get_logger().info("Loading Franka model...")
        self.robot = load_panda()

        self.get_logger().info("Creating CBF...")
        self.cbf_config = DemoConfig(self.robot, whole_body_pos_min, whole_body_pos_max)
        self.obstacle_center = self.cbf_config.collision_pos.copy()
        self.collision_pos = self.obstacle_center.copy()
        self.collision_vel = np.zeros_like(self.collision_pos)
        self.collision_acc = np.zeros_like(self.collision_pos)
        self.obstacle_travel_y = 0.40
        self.obstacle_freq = 0.25
        self.cbf = CBF.from_config(self.cbf_config)

        self.diagnostics_enabled = bool(self.get_parameter("enable_diagnostics").value)
        diagnostics_rate_hz = float(self.get_parameter("diagnostics_rate_hz").value)
        diagnostics_rate_hz = max(diagnostics_rate_hz, 1.0)
        self.log_every_n = max(1, int(round(self.control_freq / diagnostics_rate_hz)))
        self.diagnostics_path = Path("~/oscbf_diagnostics.csv").expanduser()
        self.diagnostics_file = None
        self.diagnostics_writer = None
        self.diagnostics_queue = queue.Queue(maxsize=256)
        self.diagnostics_stop = threading.Event()
        self.diagnostics_thread = None

        if self.diagnostics_enabled:
            self.diagnostics_file = self.diagnostics_path.open(
                "w", newline="", buffering=1
            )
            self.diagnostics_writer = csv.writer(self.diagnostics_file)
            self._write_diagnostics_header()
            self.diagnostics_thread = threading.Thread(
                target=self.diagnostics_worker,
                daemon=True,
            )
            self.diagnostics_thread.start()
            self.get_logger().info(
                f"Writing OSCBF diagnostics to {self.diagnostics_path} "
                f"at ~{self.control_freq / self.log_every_n:.1f} Hz"
            )

        kp_pos = 30.0
        kp_rot = 10.0
        kd_pos = 10.0
        kd_rot = 5.0
        kp_joint = 10.0
        kd_joint = 2.0

        self.osc_controller = PoseTaskTorqueController(
            n_joints=self.robot.num_joints,
            kp_task=np.concatenate([kp_pos * np.ones(3), kp_rot * np.ones(3)]),
            kd_task=np.concatenate([kd_pos * np.ones(3), kd_rot * np.ones(3)]),
            kp_joint=kp_joint,
            kd_joint=kd_joint,
            tau_min=None,
            tau_max=None,
        )

        self.get_logger().info("Jit compiling OSCBF controller...")
        self._jit_compile()

        self.get_logger().info(
            f"OSCBF Node initialization complete. startup_hold_sec={self.startup_hold_sec:.1f}"
        )
        self.get_logger().info(
            "Virtual obstacle enabled: "
            f"center={self.obstacle_center.ravel().round(4)}, "
            f"y_travel={self.obstacle_travel_y:.2f} m, "
            f"freq={self.obstacle_freq:.2f} Hz"
        )

    def _jit_compile(self):
        z_aug = np.zeros(self.robot.num_joints * 2 + 9)
        z_aug[self.robot.num_joints * 2 : self.robot.num_joints * 2 + 3] = (
            self.collision_pos.ravel()
        )
        pos = np.ones(3)
        rot = np.eye(3).ravel()
        vel = np.zeros(3)
        omega = np.zeros(3)
        z_ee_des = np.concatenate([pos, rot, vel, omega])
        _ = np.asarray(
            compute_control(
                self.robot,
                self.osc_controller,
                self.cbf,
                z_aug,
                z_ee_des,
            )[0]
        )

    def joint_state_callback(self, msg: JointState):
        self.last_joint_state = np.array([msg.position, msg.velocity]).ravel()

        if not self.initialized_desired_state:
            q = self.last_joint_state[: self.robot.num_joints]
            qdot = self.last_joint_state[self.robot.num_joints :]
            _, _, _, _, _, ee_tmat = self.robot.torque_control_matrices(q, qdot)

            pos = np.asarray(ee_tmat[:3, 3])
            rot = np.asarray(ee_tmat[:3, :3]).ravel()
            vel = np.zeros(3)
            omega = np.zeros(3)

            self.safe_desired_ee_state = np.concatenate([pos, rot, vel, omega])
            self.initialized_desired_state = True
            self.get_logger().info(
                f"Initialized desired EE pose from current robot state: pos={pos.round(4)}"
            )

    def ee_state_callback(self, msg: EEState):
        pos: Point = msg.pose.position
        quat: Quaternion = msg.pose.orientation
        vel: Vector3 = msg.twist.linear
        omega: Vector3 = msg.twist.angular

        xyzw = np.array([quat.x, quat.y, quat.z, quat.w])
        flat_rmat = xyzw_to_rotation_numpy(xyzw).flatten()

        self.last_ee_state = np.concatenate(
            [
                np.array([pos.x, pos.y, pos.z]),
                flat_rmat,
                np.array([vel.x, vel.y, vel.z]),
                np.array([omega.x, omega.y, omega.z]),
            ]
        )

    def update_virtual_obstacle(self, elapsed: float):
        omega = 2.0 * np.pi * self.obstacle_freq
        phase = omega * elapsed
        self.collision_pos = self.obstacle_center.copy()
        self.collision_vel = np.zeros_like(self.collision_pos)
        self.collision_acc = np.zeros_like(self.collision_pos)
        obstacle_amp_y = 0.5 * self.obstacle_travel_y
        self.collision_pos[0, 1] = (
            self.obstacle_center[0, 1] + obstacle_amp_y * np.sin(phase)
        )
        self.collision_vel[0, 1] = obstacle_amp_y * omega * np.cos(phase)
        self.collision_acc[0, 1] = -obstacle_amp_y * omega**2 * np.sin(phase)

    def augmented_state(self):
        return np.concatenate(
            [
                np.asarray(self.last_joint_state, dtype=float),
                self.collision_pos.ravel(),
                self.collision_vel.ravel(),
                self.collision_acc.ravel(),
            ]
        )

    def _write_diagnostics_header(self):
        header = [
            "time_sec",
            "des_ee_x",
            "des_ee_y",
            "des_ee_z",
            "follower_ee_x",
            "follower_ee_y",
            "follower_ee_z",
            "obstacle_x",
            "obstacle_y",
            "obstacle_z",
            "min_h_obstacle",
            "min_h_joint_position",
            "min_h_joint_velocity",
            "h_singularity",
            "min_h_workspace",
            "tau_correction_norm",
            "closest_sphere_x",
            "closest_sphere_y",
            "closest_sphere_z",
            "closest_sphere_radius",
            "closest_sphere_index",
        ]
        header += [f"tau_nom_{i}" for i in range(self.robot.num_joints)]
        header += [f"tau_filtered_{i}" for i in range(self.robot.num_joints)]
        header += [f"tau_cmd_{i}" for i in range(self.robot.num_joints)]
        self.diagnostics_writer.writerow(header)

    def compute_barrier_diagnostics(self, z, collision_pos):
        q = z[: self.robot.num_joints]
        qdot = z[self.robot.num_joints :]

        h_joint_position = np.concatenate(
            [
                np.asarray(self.cbf_config.q_max) - q,
                q - np.asarray(self.cbf_config.q_min),
            ]
        )

        joint_max_vels = np.asarray(self.robot.joint_max_velocities)
        h_joint_velocity = np.concatenate(
            [
                joint_max_vels - qdot,
                qdot + joint_max_vels,
            ]
        )

        sigmas = np.linalg.svd(np.asarray(self.robot.ee_jacobian(q)), compute_uv=False)
        h_singularity = float(np.prod(sigmas) - self.cbf_config.singularity_tol)

        robot_collision_pos_rad = np.asarray(self.robot.link_collision_data(q))
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3]
        robot_num_pts = robot_collision_positions.shape[0]

        whole_body_upper = (
            np.tile(self.cbf_config.whole_body_pos_max, (robot_num_pts, 1))
            - robot_collision_positions
            - robot_collision_radii[:, None]
        ).ravel()
        whole_body_lower = (
            robot_collision_positions
            - np.tile(self.cbf_config.whole_body_pos_min, (robot_num_pts, 1))
            - robot_collision_radii[:, None]
        ).ravel()
        h_workspace = np.concatenate([whole_body_upper, whole_body_lower])

        obs_pos = np.asarray(collision_pos, dtype=float)
        obs_radii = np.asarray(self.cbf_config.collision_radii, dtype=float)
        diff = robot_collision_positions[:, None, :] - obs_pos[None, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        safe_dist = (
            robot_collision_radii[:, None]
            + obs_radii[None, :]
            + self.cbf_config.collision_margin
        )
        h_obstacle = dist_sq - safe_dist**2

        closest_idx = np.unravel_index(np.argmin(h_obstacle), h_obstacle.shape)
        closest_sphere_idx = int(closest_idx[0])
        closest_sphere_pos = robot_collision_positions[closest_sphere_idx]
        closest_sphere_radius = float(robot_collision_radii[closest_sphere_idx])

        return {
            "min_h_obstacle": float(np.min(h_obstacle)),
            "min_h_joint_position": float(np.min(h_joint_position)),
            "min_h_joint_velocity": float(np.min(h_joint_velocity)),
            "h_singularity": h_singularity,
            "min_h_workspace": float(np.min(h_workspace)),
            "closest_sphere_pos": closest_sphere_pos,
            "closest_sphere_radius": closest_sphere_radius,
            "closest_sphere_index": closest_sphere_idx,
        }

    def enqueue_diagnostics(
        self,
        elapsed,
        des,
        follower_ee_pos,
        tau_nom,
        tau_filtered,
        tau_cmd,
    ):
        if not self.diagnostics_enabled:
            return

        snapshot = {
            "elapsed": float(elapsed),
            "des": np.asarray(des, dtype=float).copy(),
            "follower_ee_pos": np.asarray(follower_ee_pos, dtype=float).copy(),
            "tau_nom": np.asarray(tau_nom, dtype=float).copy(),
            "tau_filtered": np.asarray(tau_filtered, dtype=float).copy(),
            "tau_cmd": np.asarray(tau_cmd, dtype=float).copy(),
            "z": np.asarray(self.last_joint_state, dtype=float).copy(),
            "collision_pos": np.asarray(self.collision_pos, dtype=float).copy(),
        }

        try:
            self.diagnostics_queue.put_nowait(snapshot)
        except queue.Full:
            pass

    def diagnostics_worker(self):
        while not self.diagnostics_stop.is_set() or not self.diagnostics_queue.empty():
            try:
                snapshot = self.diagnostics_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.log_diagnostics_snapshot(snapshot)
            except Exception as exc:
                self.get_logger().warn(f"Diagnostics logging failed: {exc}")
            finally:
                self.diagnostics_queue.task_done()

    def log_diagnostics_snapshot(self, snapshot):
        diag = self.compute_barrier_diagnostics(
            snapshot["z"],
            snapshot["collision_pos"],
        )
        tau_nom = snapshot["tau_nom"]
        tau_filtered = snapshot["tau_filtered"]
        tau_cmd = snapshot["tau_cmd"]
        des = snapshot["des"]
        follower_ee_pos = snapshot["follower_ee_pos"]
        collision_pos = snapshot["collision_pos"]
        tau_correction_norm = float(np.linalg.norm(tau_filtered - tau_nom))
        closest_pos = diag["closest_sphere_pos"]

        row = [
            snapshot["elapsed"],
            float(des[0]),
            float(des[1]),
            float(des[2]),
            float(follower_ee_pos[0]),
            float(follower_ee_pos[1]),
            float(follower_ee_pos[2]),
            float(collision_pos[0, 0]),
            float(collision_pos[0, 1]),
            float(collision_pos[0, 2]),
            diag["min_h_obstacle"],
            diag["min_h_joint_position"],
            diag["min_h_joint_velocity"],
            diag["h_singularity"],
            diag["min_h_workspace"],
            tau_correction_norm,
            float(closest_pos[0]),
            float(closest_pos[1]),
            float(closest_pos[2]),
            diag["closest_sphere_radius"],
            diag["closest_sphere_index"],
        ]
        row += [float(x) for x in tau_nom]
        row += [float(x) for x in tau_filtered]
        row += [float(x) for x in tau_cmd]
        self.diagnostics_writer.writerow(row)

    def close_diagnostics(self):
        if not getattr(self, "diagnostics_enabled", False):
            return

        self.diagnostics_stop.set()
        if self.diagnostics_thread is not None:
            self.diagnostics_thread.join(timeout=0.5)
        if self.diagnostics_file is not None and not self.diagnostics_file.closed:
            self.diagnostics_file.close()

    def rate_limit_tau(self, tau_des, tau_prev):
        delta = tau_des - tau_prev
        delta = np.clip(delta, -self.max_delta_tau, self.max_delta_tau)
        return tau_prev + delta

    def publish_zero_torques(self):
        msg = Float64MultiArray()
        msg.data = [0.0] * self.robot.num_joints
        for _ in range(3):
            self.torque_pub.publish(msg)
            time.sleep(1 / self.control_freq)

    def publish_control(self):
        if self.last_joint_state is None or not self.initialized_desired_state:
            return

        elapsed = time.time() - self.start_time
        self.update_virtual_obstacle(elapsed)

        if elapsed < self.startup_hold_sec or self.last_ee_state is None:
            des = self.safe_desired_ee_state.copy()
        else:
            des = self.last_ee_state.copy()

        try:
            z_aug = self.augmented_state()
            tau_raw, tau_nom, tau_filtered, follower_ee_pos = compute_control(
                self.robot,
                self.osc_controller,
                self.cbf,
                z_aug,
                des,
            )
            tau = np.asarray(tau_raw)

            if not np.all(np.isfinite(tau)):
                self.get_logger().error("Non-finite torque detected. Sending zero torque.")
                self.publish_zero_torques()
                return

            tau = np.clip(tau, -self.max_abs_tau, self.max_abs_tau)
            tau = self.rate_limit_tau(tau, self.last_tau_cmd)

            if self.log_count % self.log_every_n == 0:
                self.enqueue_diagnostics(
                    elapsed,
                    des,
                    follower_ee_pos,
                    tau_nom,
                    tau_filtered,
                    tau,
                )
            self.log_count += 1

            self.last_tau_cmd = tau.copy()

            msg = Float64MultiArray()
            msg.data = tau.tolist()
            self.torque_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Control publish failed: {e}")
            self.publish_zero_torques()

    def signal_handler(self, sig, frame):
        self.get_logger().warn("Shutdown signal received, sending zero torques...")
        self.publish_zero_torques()
        self.close_diagnostics()
        self.get_logger().warn("Zero torques sent, shutting down.")
        time.sleep(0.1)
        sys.exit(0)

    def destroy_node(self):
        self.close_diagnostics()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OSCBFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {e}")
    finally:
        node.publish_zero_torques()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
