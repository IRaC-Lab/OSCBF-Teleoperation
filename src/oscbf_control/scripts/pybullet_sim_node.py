#!/usr/bin/env python3
"""Test the OSCBF ROS2 communication with pybullet

This mimics the franka c++ communication node:
- listens to joint torques
- applies torques to pybullet
- publishes joint states

Features:
- three static sphere obstacles generated inside this node
- collision detection only (penetration only)
"""

import numpy as np
import pybullet

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

from oscbf.core.manipulation_env import FrankaTorqueControlEnv


class PybulletNode(Node):
    def __init__(self):
        super().__init__("sim_node")

        self.freq = 1000
        self.env = FrankaTorqueControlEnv(timestep=1 / self.freq)

        self.obstacle_center = np.array([0.25, 0.30, 0.50], dtype=float)
        self.obstacle_amp_y = 0.1
        self.obstacle_positions = np.vstack(
            [
                self.obstacle_center + np.array([0.0, -self.obstacle_amp_y, 0.0]),
                self.obstacle_center,
                self.obstacle_center + np.array([0.0, self.obstacle_amp_y, 0.0]),
            ]
        )
        self.obstacle_radius = 0.15
        self.obstacle_rgba = [0.45, 0.75, 0.95, 0.85]
        self.obstacle_body_ids = []

        self.in_collision = False

        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.torque_sub = self.create_subscription(
            Float64MultiArray,
            "franka/torque_command",
            self.torque_callback,
            qos_profile,
        )

        self.joint_state_pub = self.create_publisher(
            JointState, "franka/joint_states", qos_profile
        )

        self._last_z = None

        self._spawn_obstacles()
        self.timer = self.create_timer(1 / self.freq, self.publish_joint_states)

        self.get_logger().info("PyBullet node initialized.")
        self.get_logger().info(
            f"Static sphere obstacles created at y={self.obstacle_positions[:, 1].round(3).tolist()}, "
            f"radius={self.obstacle_radius:.3f} m"
        )

    def _spawn_obstacles(self):
        visual_shape_id = self.env.client.createVisualShape(
            shapeType=pybullet.GEOM_SPHERE,
            radius=float(self.obstacle_radius),
            rgbaColor=self.obstacle_rgba,
        )

        collision_shape_id = self.env.client.createCollisionShape(
            shapeType=pybullet.GEOM_SPHERE,
            radius=float(self.obstacle_radius),
        )

        for obstacle_position in self.obstacle_positions:
            obstacle_body_id = self.env.client.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision_shape_id,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=obstacle_position.tolist(),
                baseOrientation=[0.0, 0.0, 0.0, 1.0],
            )
            self.obstacle_body_ids.append(obstacle_body_id)

    def get_contacts(self):
        contacts = []
        for obstacle_body_id in self.obstacle_body_ids:
            contacts.extend(
                self.env.client.getContactPoints(
                    bodyA=self.env.robot,
                    bodyB=obstacle_body_id,
                )
            )
        return contacts

    def get_penetrating_contacts(self):
        contacts = self.get_contacts()
        return [c for c in contacts if c[8] < 0.0]

    def check_collision(self) -> bool:
        return len(self.get_penetrating_contacts()) > 0

    def log_collision_once(self):
        penetrating_contacts = self.get_penetrating_contacts()
        collided = len(penetrating_contacts) > 0

        if collided and not self.in_collision:
            self.get_logger().warn("Collision detected between robot and obstacle.")
            self.get_logger().warn(
                f"Penetrating contacts: {len(penetrating_contacts)}"
            )

            for c in penetrating_contacts[:5]:
                self.get_logger().warn(
                    f"linkA={c[3]}, linkB={c[4]}, contactDistance={c[8]:.6f}, normalForce={c[9]:.6f}"
                )

        self.in_collision = collided

    def torque_callback(self, msg):
        if self._last_z is None:
            self.get_logger().warn("No joint state received yet.")
            return

        joint_torques = np.array(msg.data, dtype=float)

        zero_velocities = [0.0] * self.env.num_joints
        zero_accelerations = [0.0] * self.env.num_joints
        gravity_torques = self.env.client.calculateInverseDynamics(
            self.env.robot,
            self._last_z[: self.env.num_joints].tolist(),
            zero_velocities,
            zero_accelerations,
        )
        joint_torques += np.asarray(gravity_torques)

        self.env.apply_control(joint_torques)
        self.env.step()

    def publish_joint_states(self):
        self.log_collision_once()

        z = self.env.get_joint_state()

        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.position = z[: self.env.num_joints].tolist()
        joint_state_msg.velocity = z[self.env.num_joints :].tolist()
        self.joint_state_pub.publish(joint_state_msg)

        self._last_z = z


def main(args=None):
    rclpy.init(args=args)
    node = PybulletNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
