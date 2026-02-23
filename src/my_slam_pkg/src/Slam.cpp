#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>
#include "eufs_msgs/msg/cone_array.hpp"
#include "eufs_msgs/msg/cone_array_with_covariance.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include <vector>
#include <cmath>
#include <memory>
#include <algorithm>
#include <rclcpp/qos.hpp>

using std::placeholders::_1;

inline double wrapToPi(double a) {
    return std::atan2(std::sin(a), std::cos(a));
}

struct ConeDetection {
    double range;
    double bearing;
    int color; // 0: Blue, 1: Yellow, 2: Big Orange
};

struct LandmarkInfo {
    int color;
    int hits;
};

class EKFSLAM : public rclcpp::Node {
public:
    EKFSLAM() : Node("ekf_slam_node") {
        x_ = Eigen::VectorXd::Zero(3);
        P_ = Eigen::MatrixXd::Identity(3, 3) * 0.01;

        R_.setZero();
        R_(0, 0) = std::pow(0.5, 2); 
        R_(1, 1) = std::pow(0.1, 2);

        cones_pub_ = create_publisher<eufs_msgs::msg::ConeArray>("/planning/cones", 10);
        slam_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/slam/odom", 10);    
        
        auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
        cones_sub_ = create_subscription<eufs_msgs::msg::ConeArrayWithCovariance>(
            "/perception/cones", 
            qos, 
            std::bind(&EKFSLAM::conesCallback, this, _1));
        
        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/odometry/filtered", 10, std::bind(&EKFSLAM::odomCallback, this, _1));

        timer_ = create_wall_timer(
            std::chrono::milliseconds(50), std::bind(&EKFSLAM::runSLAM, this));

        RCLCPP_INFO(this->get_logger(), "EKF-SLAM initialized. Publishing to /slam/odom");
    }

private:
    Eigen::VectorXd x_;
    Eigen::MatrixXd P_;
    Eigen::Matrix2d R_;
    std::vector<ConeDetection> z_buffer_;
    std::vector<LandmarkInfo> lm_info_;
    bool lap_closed_ = false;
    double vx_ = 0.0, yaw_rate_ = 0.0, dt_ = 0.0;
    rclcpp::Time last_odom_time_{0, 0, RCL_ROS_TIME};

    rclcpp::Subscription<eufs_msgs::msg::ConeArrayWithCovariance>::SharedPtr cones_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<eufs_msgs::msg::ConeArray>::SharedPtr cones_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr slam_odom_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        vx_ = msg->twist.twist.linear.x;
        yaw_rate_ = msg->twist.twist.angular.z;
    }

    void conesCallback(const eufs_msgs::msg::ConeArrayWithCovariance::SharedPtr msg) {
        z_buffer_.clear();
        for (const auto &cone : msg->blue_cones) addConeToBuffer(cone.point, 0); 
        for (const auto &cone : msg->yellow_cones) addConeToBuffer(cone.point, 1);
        for (const auto &cone : msg->big_orange_cones) addConeToBuffer(cone.point, 2); 
    }

    void addConeToBuffer(const geometry_msgs::msg::Point &cone, int color) {
        if (std::abs(cone.x) > 500.0 || std::abs(cone.y) > 500.0) return;
        ConeDetection z;
        z.range = std::hypot(cone.x, cone.y);
        z.bearing = std::atan2(cone.y, cone.x);
        z.color = color;
        if (z.range > 0.1 && z.range < 30.0) z_buffer_.push_back(z);
    }

    void predict() {
        double psi = x_(2);
        x_(0) += vx_ * std::cos(psi) * dt_;
        x_(1) += vx_ * std::sin(psi) * dt_;
        x_(2) = wrapToPi(x_(2) + yaw_rate_ * dt_);

        Eigen::MatrixXd G = Eigen::MatrixXd::Identity(x_.size(), x_.size());
        G(0, 2) = -vx_ * std::sin(psi) * dt_;
        G(1, 2) = vx_ * std::cos(psi) * dt_;

        P_ = G * P_ * G.transpose();
        
        // Dynamic Q Inflation based on yaw_rate for hairpins
        double yaw_factor = std::abs(yaw_rate_) * 0.15; 
        Eigen::Matrix3d Q;
        Q << std::pow(0.1 + yaw_factor, 2), 0, 0, 
             0, std::pow(0.1 + yaw_factor, 2), 0, 
             0, 0, std::pow(0.05 + yaw_factor, 2);
             
        P_.block<3, 3>(0, 0) += Q;
    }

    void measurementModel(int j, Eigen::Vector2d &zhat, Eigen::MatrixXd &H) {
        int idx = 3 + 2 * j;
        double dx = x_(idx) - x_(0), dy = x_(idx + 1) - x_(1);
        double r2 = dx * dx + dy * dy, r = std::sqrt(r2) + 1e-6;
        zhat << r, wrapToPi(std::atan2(dy, dx) - x_(2));
        H = Eigen::MatrixXd::Zero(2, x_.size());
        H(0, 0) = -dx / r;  H(0, 1) = -dy / r;  H(0, 2) = 0;
        H(1, 0) = dy / r2;  H(1, 1) = -dx / r2; H(1, 2) = -1.0;
        H(0, idx) = dx / r; H(0, idx + 1) = dy / r;
        H(1, idx) = -dy / r2; H(1, idx + 1) = dx / r2;
    }

    void runSLAM() {
        rclcpp::Time current_time = this->now();
        if (last_odom_time_.nanoseconds() == 0) {
            last_odom_time_ = current_time;
            return;
        }
        dt_ = (current_time - last_odom_time_).seconds();
        last_odom_time_ = current_time;
        
        if (dt_ <= 0.0 || dt_ > 0.2) return;
        predict();

        for (const auto &z : z_buffer_) {
            int best_j = -1; double best_md = 1e9;
            for (size_t j = 0; j < lm_info_.size(); ++j) {
                if (lm_info_[j].color != z.color) continue;
                Eigen::Vector2d zhat; Eigen::MatrixXd H;
                measurementModel(j, zhat, H);
                Eigen::Vector2d v; v << z.range - zhat(0), wrapToPi(z.bearing - zhat(1));
                Eigen::Matrix2d S = H * P_ * H.transpose() + R_;
                
                if (S.determinant() < 1e-6) continue; 

                double md = v.transpose() * S.inverse() * v;
                if (md < best_md && md < 9.21) { best_md = md; best_j = (int)j; }
            }

            if (best_j >= 0) {
                lm_info_[best_j].hits++;
                Eigen::Vector2d zhat; Eigen::MatrixXd H;
                measurementModel(best_j, zhat, H);
                Eigen::Vector2d v; v << z.range - zhat(0), wrapToPi(z.bearing - zhat(1));
                Eigen::Matrix2d S = H * P_ * H.transpose() + R_;
                Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();
                x_ += K * v;
                P_ = (Eigen::MatrixXd::Identity(x_.size(), x_.size()) - K * H) * P_;
            } else if (!lap_closed_) {
                double angle = wrapToPi(x_(2) + z.bearing);
                double lx = x_(0) + z.range * std::cos(angle), ly = x_(1) + z.range * std::sin(angle);
                bool duplicate = false;
                for(size_t i=0; i < lm_info_.size(); ++i){
                    if(std::hypot(x_(3+2*i)-lx, x_(3+2*i+1)-ly) < 2.0) { duplicate = true; break; }
                }
                if(!duplicate && lm_info_.size() < 400) {
                    int old_size = x_.size();
                    x_.conservativeResize(old_size + 2); x_.tail<2>() << lx, ly;
                    Eigen::MatrixXd Gx(2, 3); Gx << 1, 0, -z.range*std::sin(angle), 0, 1, z.range*std::cos(angle);
                    Eigen::Matrix2d Gz; Gz << std::cos(angle), -z.range*std::sin(angle), std::sin(angle), z.range*std::cos(angle);
                    Eigen::MatrixXd P_new = Eigen::MatrixXd::Zero(old_size+2, old_size+2);
                    P_new.topLeftCorner(old_size, old_size) = P_;
                    Eigen::MatrixXd Pxr = P_.block(0, 0, old_size, 3) * Gx.transpose();
                    P_new.block(0, old_size, old_size, 2) = Pxr; P_new.block(old_size, 0, 2, old_size) = Pxr.transpose();
                    P_new.bottomRightCorner(2, 2) = Gx * P_.block<3, 3>(0, 0) * Gx.transpose() + Gz * R_ * Gz.transpose();
                    P_ = P_new; lm_info_.push_back({z.color, 1});
                }
            }
        }
        z_buffer_.clear();
        P_ = 0.5 * (P_ + P_.transpose()); P_ += Eigen::MatrixXd::Identity(P_.rows(), P_.cols()) * 1e-9;

        // Publish Cones
        auto out_msg = eufs_msgs::msg::ConeArray();
        out_msg.header.stamp = this->now();
        out_msg.header.frame_id = "map";
        for (size_t i = 0; i < lm_info_.size(); ++i) {
            geometry_msgs::msg::Point p; p.x = x_(3 + 2 * i); p.y = x_(3 + 2 * i + 1); p.z = 0.0;
            if (lm_info_[i].color == 0) out_msg.blue_cones.push_back(p);
            else if (lm_info_[i].color == 1) out_msg.yellow_cones.push_back(p);
            else if (lm_info_[i].color == 2) out_msg.big_orange_cones.push_back(p);
        }
        cones_pub_->publish(out_msg);

        // Publish SLAM Odometry for the Controller
        nav_msgs::msg::Odometry slam_odom;
        slam_odom.header.stamp = this->now();
        slam_odom.header.frame_id = "map";
        
        slam_odom.pose.pose.position.x = x_(0);
        slam_odom.pose.pose.position.y = x_(1);
        slam_odom.pose.pose.position.z = 0.0;
        
        slam_odom.pose.pose.orientation.x = 0.0; 
        slam_odom.pose.pose.orientation.y = 0.0;
        slam_odom.pose.pose.orientation.z = std::sin(x_(2) * 0.5); 
        slam_odom.pose.pose.orientation.w = std::cos(x_(2) * 0.5);
        
        slam_odom.twist.twist.linear.x = vx_;
        slam_odom.twist.twist.angular.z = yaw_rate_;
        
        slam_odom_pub_->publish(slam_odom);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EKFSLAM>());
    rclcpp::shutdown();
    return 0;
}