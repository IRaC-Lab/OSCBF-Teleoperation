#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <Eigen/Dense>
#include <franka/robot.h>
#include <franka/model.h>

#include <thread>
#include <atomic>
#include <mutex>
#include <string.h>

using namespace std::chrono_literals;

/**
 * Sets a default collision behavior, joint impedance and Cartesian impedance.
 *
 * @param[in] robot Robot instance to set behavior on.
 */
void setDefaultBehavior(franka::Robot& robot) {
  robot.setCollisionBehavior(
      {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}}, {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
      {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}}, {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}},
      {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}}, {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
      {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0}}, {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0}});
  robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
  robot.setCartesianImpedance({{3000, 3000, 3000, 300, 300, 300}});
}

/**
 * @class FrankaImpedanceController
 * @brief A ROS2 node for controlling a Franka Panda robot using torque control
 */
class FrankaImpedanceController : public rclcpp::Node {
public:
  FrankaImpedanceController()
      : Node("franka_impedance_controller"), stop_control_loop_(false) {
    RCLCPP_INFO(this->get_logger(), "Starting Controller...");

    this->declare_parameter<std::string>("robot_hostname", "172.16.0.2");
    this->declare_parameter<bool>("realtime", false);

    rclcpp::QoS qos(rclcpp::KeepLast(10));
    qos.best_effort();
    qos.durability_volatile();

    torque_subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
        "franka/torque_command", qos,
        std::bind(&FrankaImpedanceController::torqueCallback, this, std::placeholders::_1));

    joint_state_publisher_ =
        this->create_publisher<sensor_msgs::msg::JointState>("franka/joint_states", qos);

    ee_pose_publisher_ =
        this->create_publisher<geometry_msgs::msg::PoseStamped>("franka/ee_pose", qos);

    joint_state_timer_ =
        this->create_wall_timer(1ms, std::bind(&FrankaImpedanceController::publishJointStates, this));

    std::string robot_hostname = this->get_parameter("robot_hostname").as_string();

    try {
      if (this->get_parameter("realtime").as_bool()) {
        robot_ = std::make_unique<franka::Robot>(robot_hostname, franka::RealtimeConfig::kEnforce);
      } else {
        robot_ = std::make_unique<franka::Robot>(robot_hostname, franka::RealtimeConfig::kIgnore);
      }
      RCLCPP_INFO(this->get_logger(), "Robot connected: %s", robot_hostname.c_str());
      // setDefaultBehavior(*robot_);
      // model_ = std::make_unique<franka::Model>(robot_->loadModel());
      // RCLCPP_INFO(this->get_logger(), "Model loaded.");
    } catch (const std::exception& ex) {
      RCLCPP_ERROR(this->get_logger(), "Failed to connect to robot: %s", ex.what());
    }

    if (robot_) {
      last_robot_state_ = robot_->readOnce();
    }

    try {
      control_thread_ = std::thread(&FrankaImpedanceController::controlLoop, this);
    } catch (const std::exception& ex) {
      RCLCPP_ERROR(this->get_logger(), "Failed to start control loop: %s", ex.what());
    }
  }

  ~FrankaImpedanceController() {
    stop_control_loop_ = true;
    if (control_thread_.joinable()) {
      control_thread_.join();
    }
  }

private:
  void publishJointStates() {
    if (!robot_) {
      RCLCPP_WARN(this->get_logger(), "Robot not initialized");
      return;
    }

    try {
      franka::RobotState current_state;
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        current_state = last_robot_state_;
      }

      sensor_msgs::msg::JointState joint_state_msg;
      joint_state_msg.header.stamp = this->now();
      joint_state_msg.name = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"};
      joint_state_msg.position.assign(current_state.q.begin(), current_state.q.end());
      joint_state_msg.velocity.assign(current_state.dq.begin(), current_state.dq.end());
      joint_state_publisher_->publish(joint_state_msg);

      Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::ColMajor>> T(current_state.O_T_EE.data());
      Eigen::Matrix3d R = T.block<3, 3>(0, 0);
      Eigen::Quaterniond quat(R);
      quat.normalize();

      geometry_msgs::msg::PoseStamped ee_pose_msg;
      ee_pose_msg.header.stamp = joint_state_msg.header.stamp;
      ee_pose_msg.header.frame_id = "panda_link0";

      ee_pose_msg.pose.position.x = T(0, 3);
      ee_pose_msg.pose.position.y = T(1, 3);
      ee_pose_msg.pose.position.z = T(2, 3);

      ee_pose_msg.pose.orientation.x = quat.x();
      ee_pose_msg.pose.orientation.y = quat.y();
      ee_pose_msg.pose.orientation.z = quat.z();
      ee_pose_msg.pose.orientation.w = quat.w();

      ee_pose_publisher_->publish(ee_pose_msg);

    } catch (const std::exception& ex) {
      RCLCPP_ERROR(this->get_logger(), "Joint/EE state update error: %s", ex.what());
    }
  }

  void torqueCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    if (!robot_) {
      RCLCPP_WARN(this->get_logger(), "Robot not initialized");
      return;
    }

    if (msg->data.size() < 7) {
      RCLCPP_ERROR(this->get_logger(), "Received torque command with insufficient data");
      return;
    }

    std::lock_guard<std::mutex> lock(torque_mutex_);
    std::copy(msg->data.begin(), msg->data.begin() + 7, tau_command_.begin());
  }

  void controlLoop() {
    if (!robot_) {
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Starting impedance control loop...");

    try {
      robot_->control([this](const franka::RobotState& state, franka::Duration) -> franka::Torques {
        if (stop_control_loop_) {
          RCLCPP_INFO(this->get_logger(), "Stopping control loop...");
          franka::Torques zero_torque = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
          return franka::MotionFinished(zero_torque);
        }

        {
          std::lock_guard<std::mutex> lock(state_mutex_);
          last_robot_state_ = state;
        }

        std::array<double, 7> tau;
        {
          std::lock_guard<std::mutex> lock(torque_mutex_);
          tau = tau_command_;
        }

        return tau;
      });

    } catch (const std::exception& ex) {
      RCLCPP_ERROR(this->get_logger(), "Control loop error: %s", ex.what());
    }
  }

  std::unique_ptr<franka::Robot> robot_;
  // std::unique_ptr<franka::Model> model_;

  franka::RobotState last_robot_state_;

  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr torque_subscription_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr ee_pose_publisher_;
  rclcpp::TimerBase::SharedPtr joint_state_timer_;

  std::thread control_thread_;
  std::atomic<bool> stop_control_loop_;
  std::mutex torque_mutex_;
  std::mutex state_mutex_;
  std::array<double, 7> tau_command_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FrankaImpedanceController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
