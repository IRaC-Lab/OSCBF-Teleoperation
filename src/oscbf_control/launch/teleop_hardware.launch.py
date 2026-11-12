from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="oscbf_control",
                executable="franka_impedance_controller",
                name="franka_impedance_controller",
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="oscbf_control",
                executable="franka_control_node.py",
                name="control_node",
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="oscbf_control",
                executable="traj_node.py",
                name="ee_traj_node",
                output="screen",
                emulate_tty=True,
            ),
        ]
    )
