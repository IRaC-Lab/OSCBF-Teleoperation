# OSCBF-Teleoperation

[![Paper](http://img.shields.io/badge/arXiv-2503.06736-B31B1B.svg)](https://arxiv.org/abs/2503.06736)

Fast, safe manipulator teleoperation with [OSCBF](https://github.com/StanfordASL/oscbf)

Currently supported hardware platforms:
- Franka Emika Panda

## Installation

### Libfranka

For newer robots, you can probably follow the standard setup details on the [libfranka Github](https://github.com/frankarobotics/libfranka). However, our lab has an older Panda, which requires libfranka 0.8.0. To get this to work, I had to make a minor change to libfranka which is available [here](https://github.com/danielpmorton/libfranka_08_patch).

### Fetch the code

Clone the original oscbf code and the ROS2 workspace. See the README on the [OSCBF Github](https://github.com/StanfordASL/oscbf) for additional details
```
git clone https://github.com/StanfordASL/oscbf
git clone https://github.com/StanfordASL/oscbf_hardware_ws/
```

### Set up your virtual environment

If using a virtual environment, keep in mind that while the OSCBF code runs on multiple python versions (tested: 3.10, 3.11, 3.12), to get it to work with ROS2, you'll need the python version to match with your ROS2 python version. For ROS2 Humble, this means 3.10.x, and for ROS2 Jazzy, this means 3.12.x. 

I recommend `uv` to manage the virtual environment. If you don't have `uv` already installed, run the following:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
# Optional, but recommended:
# echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
```
Then create the virtual environment for the project and install dependencies
```
cd oscbf_hardware_ws
uv venv --python 3.12.3 --system-site-packages
# Or, feel free to select a different python version.
# If you intend to use ROS2 Humble, I recommend 3.10.12
source .venv/bin/activate
# Note: this assumes you cloned oscbf and oscbf_hardware_ws in the same directory
cd ../oscbf
uv pip install -e .
```

### Oculus Reader

If using the Oculus, follow the steps as found on the [oculus_reader github page](https://github.com/rail-berkeley/oculus_reader). Then, install the package in your environment with
```
source oscbf_hardware_ws/.venv/bin/activate
cd oculus_reader # Wherever you cloned it
uv pip install -e .
```

When working with the Oculus/Quest hardware, the following tips might be useful:
- Go into settings and turn all of the automatic sleep times to the maximum value (4 hours)
- Add a sticker on top of the proximity sensor on the inside of the headset
- The Meta Quest 3 sometimes has some issues where it loses track of the controller, and then when it regains tracking, it "snaps" to the new location, leading to unstable robot control. The Quest 2 seems to be more stable here.

### Build + Setup

To build, run
```
./scripts/build.sh
```
To configure your terminal/environment, run
```
source scripts/setup.sh
```

## Demo (ROS-free)

There are many interactive demos available in the original OSCBF repo! Give these a try and make sure that things work before trying to run the ROS2 nodes.

## ROS Demo

For a ROS2 end-effector trajectory tracking demo in sim,
```
cd oscbf_hardware_ws
source scripts/setup.sh
ros2 launch oscbf_control traj_sim.launch.py
```

## Overview

This contains two packages:

1. `oscbf_control`: Control nodes (Python and C++)
3. `oscbf_control_msgs`: Custom ROS2 message definitions

## Running after connecting OMX teleoperation

After connecting OMX teleoperation, start the following three nodes in order, each in a separate terminal.
OMX acts as the leader and Franka as the follower. Configure the workspace environment in each terminal as shown below.

**Terminal 1 (Franka node)**: Publishes joint states, subscribes to joint torques
```
cd oscbf_hardware_ws
source scripts/setup.sh
ros2 run oscbf_control franka_impedance_controller
```

**Terminal 2 (OSCBF node)**: Publishes joint torques, subscribes to joint states and desired EE state
```
cd oscbf_hardware_ws
source scripts/setup.sh
ros2 run oscbf_control franka_control_node.py
```

**Terminal 3 (OMX leader-following node)**: Reads the OMX end-effector TF and publishes the desired pose and twist for Franka.
```
cd oscbf_hardware_ws
source scripts/setup.sh
ros2 run oscbf_control traj_node.py
```

### Alternative terminal setup options

#### Oculus teleoperation

To use Oculus instead of OMX, replace Terminal 3 with the following node.
```
cd oscbf_hardware_ws
source scripts/setup.sh
ros2 run oscbf_control oculus_node.py
```

#### Testing in simulation

Terminal 1 can be replaced with a simulated pybullet environment which does not require a connection to the robot, and can be used to debug the controller prior to testing on hardware
```
cd oscbf_hardware_ws
source scripts/setup.sh
ros2 run oscbf_control pybullet_sim_node.py
```

## Assorted notes

- Connect to the Franka control box via ethernet
- Make sure that the ethernet profile is configured to Franka (see the Setting Up the Network section of [Franka FCI documentation](https://frankarobotics.github.io/docs/getting_started.html) if this is not already configured)
- Make sure that the robot joints are unlocked (accessed via the [Franka Desk](https://172.16.0.2/desk/)) and that the emergency stop button is not depressed. The robot should be in the blue light mode to begin accepting commands over FCI. This code also currently assumes that the Franka hand is attached.
- If the Franka node (Terminal 1) reports an error like `Move command rejected: command not possible in the current mode!`, depress and release the emergency stop to reset the mode. The robot should return to the blue light state and you can re-run the command

## Citation
```
@inproceedings{morton2025oscbf,
  author={Morton, Daniel and Pavone, Marco},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions}, 
  year={2025},
  pages={187-194},
  doi={10.1109/IROS60139.2025.11246389}
}
```
