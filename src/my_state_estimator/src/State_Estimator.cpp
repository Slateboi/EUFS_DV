#include <memory>
#include <functional>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <eigen3/Eigen/Dense>

using namespace std::chrono_literals;
using std::placeholders::_1;

struct VehicleParams {
    double Lf = 1.6; double Lr = 1.4; double w = 1.2;
    double Iz = 300.0; double R = 0.23;
    double C_alpha_f = 60000.0; double C_alpha_r = 60000.0;
};

class StateEstimator : public rclcpp::Node {
public:
    StateEstimator() : Node("state_estimator") {
        x_hat_ = Eigen::VectorXd::Zero(12);
        P_ = Eigen::MatrixXd::Identity(12, 12) * 1.0;
        u_ = Eigen::VectorXd::Zero(3);
        y_ = Eigen::VectorXd::Zero(7);
        Q_ = Eigen::MatrixXd::Identity(12, 12) * 0.01;
        R_mat_ = Eigen::MatrixXd::Identity(7, 7) * 2.0;

        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu/data", 10, std::bind(&StateEstimator::imuCallback, this, _1));
        joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10, std::bind(&StateEstimator::jointCallback, this, _1));
        
        state_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odometry/filtered", 10);
        timer_ = this->create_wall_timer(10ms, std::bind(&StateEstimator::runEKF, this));
        
        RCLCPP_INFO(this->get_logger(), "State Estimator with NaN Protection Online.");
    }

private:
    VehicleParams p_; 
    Eigen::VectorXd x_hat_, u_, y_;
    Eigen::MatrixXd P_, Q_, R_mat_;
    double global_x_, global_y_, global_yaw_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr state_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        y_(0) = msg->angular_velocity.z;
        y_(1) = msg->linear_acceleration.x;
        y_(2) = msg->linear_acceleration.y;
    }

    void jointCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        for (size_t i = 0; i < msg->name.size(); i++) {
            if (msg->name[i] == "wheel_fl_joint") y_(3) = msg->velocity[i];
            else if (msg->name[i] == "wheel_fr_joint") y_(4) = msg->velocity[i];
        }
    }

    Eigen::VectorXd vehicleProcessModel(const Eigen::VectorXd &x, double dt) {
        Eigen::VectorXd x_next = x;
        double vx = std::max(x(0), 0.1); // Avoid div by zero
        double vy = x(1); double r = x(2); double ax = x(3);
        x_next(0) += dt * (ax + r * vy);
        x_next(1) += dt * (x(4) - r * vx);
        x_next(2) += dt * (((p_.Lf * (-p_.C_alpha_f * (u_(2) - std::atan2(vy + p_.Lf*r, vx)))) - 
                           (p_.Lr * (-p_.C_alpha_r * (-std::atan2(vy - p_.Lr*r, vx))))) / p_.Iz);
        return x_next;
    }

    void runEKF() {
        // 1. Prediction
        Eigen::VectorXd x_pred = vehicleProcessModel(x_hat_, 0.01);
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(12, 12);
        Eigen::MatrixXd P_pred = F * P_ * F.transpose() + Q_;

        // 2. Update with Singularity Guard
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(7, 12); H.block<7,7>(0,0) = Eigen::MatrixXd::Identity(7,7);
        Eigen::MatrixXd S = H * P_pred * H.transpose() + R_mat_;
        
        // Use Pseudo-Inverse or robust solver to avoid NaN if S is singular
        Eigen::MatrixXd K = P_pred * H.transpose() * S.ldlt().solve(Eigen::MatrixXd::Identity(7,7));

        Eigen::VectorXd innovation = y_ - (H * x_pred);
        x_hat_ = x_pred + K * innovation;
        P_ = (Eigen::MatrixXd::Identity(12, 12) - K * H) * P_pred;

        // 3. Global NaN Reset Trigger
        if (std::isnan(x_hat_(0)) || std::isnan(global_x_)) {
            RCLCPP_ERROR(this->get_logger(), "EKF Diverged! Resetting state.");
            x_hat_.setZero();
            global_x_ = 0.0; global_y_ = 0.0;
            P_ = Eigen::MatrixXd::Identity(12, 12) * 1.0;
        }

        // 4. Numerical Stability
        P_ = 0.5 * (P_ + P_.transpose());
        P_ += Eigen::MatrixXd::Identity(12, 12) * 1e-9;

        global_yaw_ += x_hat_(2) * 0.01;
        global_x_ += (x_hat_(0) * cos(global_yaw_) - x_hat_(1) * sin(global_yaw_)) * 0.01;
        global_y_ += (x_hat_(0) * sin(global_yaw_) + x_hat_(1) * cos(global_yaw_)) * 0.01;

        nav_msgs::msg::Odometry odom;
        odom.header.stamp = this->now();
        odom.header.frame_id = "map";
        odom.pose.pose.position.x = global_x_; odom.pose.pose.position.y = global_y_;
        tf2::Quaternion q; q.setRPY(0, 0, global_yaw_);
        odom.pose.pose.orientation.x = q.x(); odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z(); odom.pose.pose.orientation.w = q.w();
        odom.twist.twist.linear.x = x_hat_(0); odom.twist.twist.angular.z = x_hat_(2);
        state_pub_->publish(odom);
    }
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StateEstimator>());
    rclcpp::shutdown();
    return 0;
}