#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from rclpy.time import Time

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from sensor_msgs.msg import JointState
from oscbf_control_msgs.msg import EEState
from geometry_msgs.msg import Point, Quaternion, Vector3


class EETrajNode(Node):
    def __init__(self):
        super().__init__("ee_traj_node")

        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("leader_base_frame", "world")
        self.declare_parameter("leader_ee_frame", "end_effector_link")
        self.declare_parameter("ee_state_topic", "/ee_state")
        self.declare_parameter("estimate_velocity", True)
        self.declare_parameter("max_dt", 0.2)

        self.declare_parameter("startup_hold_sec", 1.0)
        self.declare_parameter("max_pos_step", 0.005)
        self.declare_parameter("max_rot_step_rad", 0.05)
        self.declare_parameter("max_lin_vel", 0.25)
        self.declare_parameter("max_ang_vel", 1.0)

        self.declare_parameter("scale_x", 2.25)
        self.declare_parameter("scale_y", 2.25)
        self.declare_parameter("scale_z", 4.793)
        self.declare_parameter("offset_z", -0.063)

        self.joint_state_topic = (
            self.get_parameter("joint_state_topic").get_parameter_value().string_value
        )
        self.leader_base_frame = (
            self.get_parameter("leader_base_frame").get_parameter_value().string_value
        )
        self.leader_ee_frame = (
            self.get_parameter("leader_ee_frame").get_parameter_value().string_value
        )
        self.ee_state_topic = (
            self.get_parameter("ee_state_topic").get_parameter_value().string_value
        )
        self.estimate_velocity = bool(self.get_parameter("estimate_velocity").value)
        self.max_dt = float(self.get_parameter("max_dt").value)

        self.startup_hold_sec = float(self.get_parameter("startup_hold_sec").value)
        self.max_pos_step = float(self.get_parameter("max_pos_step").value)
        self.max_rot_step_rad = float(self.get_parameter("max_rot_step_rad").value)
        self.max_lin_vel = float(self.get_parameter("max_lin_vel").value)
        self.max_ang_vel = float(self.get_parameter("max_ang_vel").value)

        self.scale_x = float(self.get_parameter("scale_x").value)
        self.scale_y = float(self.get_parameter("scale_y").value)
        self.scale_z = float(self.get_parameter("scale_z").value)
        self.offset_z = float(self.get_parameter("offset_z").value)

        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.ee_state_pub = self.create_publisher(
            EEState, self.ee_state_topic, qos_profile
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_state_topic,
            self.joint_state_callback,
            qos_profile,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.prev_pub_pos = None
        self.prev_pub_rot = None
        self.prev_pub_t = None

        self.hold_pos = None
        self.hold_rot = None

        self.warned_tf = False
        self.start_time = self.get_clock().now()
        self.debug_count = 0

        self.get_logger().info("EE absolute leader-following node initialized.")
        self.get_logger().info(f"Listening joint states: {self.joint_state_topic}")
        self.get_logger().info(
            f"Reading TF: {self.leader_base_frame} -> {self.leader_ee_frame}"
        )
        self.get_logger().info(f"Publishing EEState: {self.ee_state_topic}")

    def joint_state_callback(self, msg: JointState):
        now = self.get_clock().now()

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.leader_base_frame,
                self.leader_ee_frame,
                Time(),
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            if not self.warned_tf:
                self.get_logger().warn(
                    f"TF lookup failed: {self.leader_base_frame} -> {self.leader_ee_frame}"
                )
                self.warned_tf = True
            return

        self.warned_tf = False

        des_pos, des_rot = self.read_leader_pose(tf_msg)

        if self.hold_pos is None:
            self.hold_pos = des_pos.copy()
            self.hold_rot = des_rot.copy()
            self.prev_pub_pos = des_pos.copy()
            self.prev_pub_rot = des_rot.copy()
            self.prev_pub_t = now.nanoseconds / 1e9
            self.get_logger().info(
                f"Initialized hold pose at {np.round(self.hold_pos, 4)}"
            )

        elapsed = (now - self.start_time).nanoseconds / 1e9
        if elapsed < self.startup_hold_sec:
            des_pos = self.hold_pos.copy()
            des_rot = self.hold_rot.copy()

        des_pos, des_rot = self.apply_rate_limits(des_pos, des_rot)

        des_vel = np.zeros(3)
        des_omega = np.zeros(3)

        now_sec = now.nanoseconds / 1e9
        if self.estimate_velocity and self.prev_pub_t is not None:
            dt = now_sec - self.prev_pub_t
            if 1e-6 < dt < self.max_dt:
                des_vel = (des_pos - self.prev_pub_pos) / dt
                des_vel = np.clip(des_vel, -self.max_lin_vel, self.max_lin_vel)

                R_delta = self.prev_pub_rot.T @ des_rot
                rotvec = rotation_matrix_to_rotvec(R_delta)
                des_omega = rotvec / dt
                des_omega = np.clip(des_omega, -self.max_ang_vel, self.max_ang_vel)

        des_quat = rmat_to_quat_xyzw(des_rot)

        out = EEState()
        out.header.stamp = now.to_msg()
        out.pose.position = Point(
            x=float(des_pos[0]),
            y=float(des_pos[1]),
            z=float(des_pos[2]),
        )
        out.pose.orientation = Quaternion(
            x=float(des_quat[0]),
            y=float(des_quat[1]),
            z=float(des_quat[2]),
            w=float(des_quat[3]),
        )
        out.twist.linear = Vector3(
            x=float(des_vel[0]),
            y=float(des_vel[1]),
            z=float(des_vel[2]),
        )
        out.twist.angular = Vector3(
            x=float(des_omega[0]),
            y=float(des_omega[1]),
            z=float(des_omega[2]),
        )

        self.debug_count += 1
        if self.debug_count % 100 == 0:
            self.get_logger().info(
                f"des_pos={np.round(des_pos, 4)}, des_vel={np.round(des_vel, 4)}"
            )

        self.ee_state_pub.publish(out)

        self.prev_pub_pos = des_pos.copy()
        self.prev_pub_rot = des_rot.copy()
        self.prev_pub_t = now_sec

    def read_leader_pose(self, tf_msg):
        tx = tf_msg.transform.translation.x
        ty = tf_msg.transform.translation.y
        tz = tf_msg.transform.translation.z

        pose_x = tx * self.scale_x
        pose_y = ty * self.scale_y
        pose_z = tz * self.scale_z + self.offset_z
        leader_pos = np.array([pose_x, pose_y, pose_z], dtype=float)

        qx = tf_msg.transform.rotation.x
        qy = tf_msg.transform.rotation.y
        qz = tf_msg.transform.rotation.z
        qw = tf_msg.transform.rotation.w

        leader_quat = np.array([qx, qy, qz, qw], dtype=float)
        r1 = quat_to_rmat_xyzw(leader_quat)

        A = np.array(
            [
                [0, 0, 1],
                [0, -1, 0],
                [1, 0, 0],
            ],
            dtype=float,
        )

        leader_rot = r1 @ A
        return leader_pos, leader_rot

    def apply_rate_limits(self, des_pos, des_rot):
        if self.prev_pub_pos is None or self.prev_pub_rot is None:
            return des_pos, des_rot

        delta_pos = des_pos - self.prev_pub_pos
        delta_norm = np.linalg.norm(delta_pos)
        if delta_norm > self.max_pos_step and delta_norm > 1e-9:
            delta_pos = delta_pos * (self.max_pos_step / delta_norm)
        pos_limited = self.prev_pub_pos + delta_pos

        R_delta = self.prev_pub_rot.T @ des_rot
        rotvec = rotation_matrix_to_rotvec(R_delta)
        angle = np.linalg.norm(rotvec)

        if angle > self.max_rot_step_rad and angle > 1e-9:
            rotvec = rotvec * (self.max_rot_step_rad / angle)
            rot_limited = self.prev_pub_rot @ rotvec_to_rotation_matrix(rotvec)
        else:
            rot_limited = des_rot

        return pos_limited, rot_limited


def quat_to_rmat_xyzw(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=float)
    q = q / np.linalg.norm(q)
    x, y, z, w = q

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=float,
    )


def rmat_to_quat_xyzw(rmat: np.ndarray) -> np.ndarray:
    tr = rmat[0, 0] + rmat[1, 1] + rmat[2, 2]

    if tr >= 0:
        s4 = 2.0 * np.sqrt(tr + 1.0)
        x = (rmat[2, 1] - rmat[1, 2]) / s4
        y = (rmat[0, 2] - rmat[2, 0]) / s4
        z = (rmat[1, 0] - rmat[0, 1]) / s4
        w = s4 / 4.0
    elif rmat[0, 0] > rmat[1, 1] and rmat[0, 0] > rmat[2, 2]:
        s4 = 2.0 * np.sqrt(1.0 + rmat[0, 0] - rmat[1, 1] - rmat[2, 2])
        x = s4 / 4.0
        y = (rmat[0, 1] + rmat[1, 0]) / s4
        z = (rmat[2, 0] + rmat[0, 2]) / s4
        w = (rmat[2, 1] - rmat[1, 2]) / s4
    elif rmat[1, 1] > rmat[2, 2]:
        s4 = 2.0 * np.sqrt(1.0 + rmat[1, 1] - rmat[0, 0] - rmat[2, 2])
        x = (rmat[0, 1] + rmat[1, 0]) / s4
        y = s4 / 4.0
        z = (rmat[1, 2] + rmat[2, 1]) / s4
        w = (rmat[0, 2] - rmat[2, 0]) / s4
    else:
        s4 = 2.0 * np.sqrt(1.0 + rmat[2, 2] - rmat[0, 0] - rmat[1, 1])
        x = (rmat[2, 0] + rmat[0, 2]) / s4
        y = (rmat[1, 2] + rmat[2, 1]) / s4
        z = s4 / 4.0
        w = (rmat[1, 0] - rmat[0, 1]) / s4

    quat = np.array([x, y, z, w], dtype=float)
    quat /= np.linalg.norm(quat)
    return quat


def rotation_matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.arccos(cos_theta)

    if theta < 1e-9:
        return np.zeros(3)

    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=float,
    )
    axis = axis / (2.0 * np.sin(theta))
    return axis * theta


def rotvec_to_rotation_matrix(rotvec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return np.eye(3)

    axis = rotvec / theta
    x, y, z = axis

    K = np.array(
        [
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0],
        ],
        dtype=float,
    )

    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def main(args=None):
    rclpy.init(args=args)
    node = EETrajNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()