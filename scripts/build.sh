#!/bin/bash
#
# This script builds all ROS2 packages

set -e

WS_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
VENV_DIR="$WS_ROOT/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: Could not find the virtual env"
  echo "Check that you've set up the venv with UV"
  exit 1
fi

cd "$WS_ROOT"
source "$VENV_DIR/bin/activate"

PYTHON_EXECUTABLE=$(which python3)
echo "Using Python: $PYTHON_EXECUTABLE"

source /opt/ros/humble/setup.bash

colcon build --symlink-install \
    --cmake-args -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE

echo "Build complete. Use 'source scripts/setup.sh' to refresh your environment."
