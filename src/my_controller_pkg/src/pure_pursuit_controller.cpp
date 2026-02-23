#include <memory>
#include <functional>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <visualization_msgs/msg/marker.hpp>

using std::placeholders::_1;

struct StateEstimate {
    double x = 0.0, y = 0.0, psi = 0.0, vx = 0.0;
};

class PurePursuitNode : public rclcpp::Node {
public:
    PurePursuitNode() : Node("pure_pursuit_node") {
        // --- TUNING PARAMETERS ---
        this->declare_parameter("L_base", 1.53); // Wheelbase of the car
        this->declare_parameter("L_min", 3.0);   // Minimum lookahead distance (shrink this for tighter hairpins)
        this->declare_parameter("k_pure", 0.8);  // Velocity multiplier for lookahead
        this->declare_parameter("delta_max", 0.45); // Max steering angle (radians)
        this->declare_parameter("target_velocity", 3.0); // Target speed

        // 1. FIXED: Listen to SLAM's corrected pose so the path and car don't drift apart!
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/slam/odom", 10, std::bind(&PurePursuitNode::odomCallback, this, _1));
        
        path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/target_path", 10, std::bind(&PurePursuitNode::pathCallback, this, _1));

        drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            "/cmd", 10);
            
        vis_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("lookahead_marker", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50), std::bind(&PurePursuitNode::controlLoop, this));
            
        RCLCPP_INFO(this->get_logger(), "Pure Pursuit Controller Online. Listening to /slam/odom");
    }

private:
    StateEstimate est_;
    nav_msgs::msg::Path path_;
    bool has_odom_ = false;
    bool has_path_ = false;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr vis_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        est_.x = msg->pose.pose.position.x;
        est_.y = msg->pose.pose.position.y;
        est_.vx = msg->twist.twist.linear.x;

        tf2::Quaternion q(
            msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
        tf2::Matrix3x3 m(q);
        double r, p, yaw;
        m.getRPY(r, p, yaw);
        est_.psi = yaw;
        has_odom_ = true;
    }

    void pathCallback(const nav_msgs::msg::Path::SharedPtr msg) {
        if (msg->poses.empty()) return;
        path_ = *msg;
        has_path_ = true;
    }

    void controlLoop() {
        if (!has_odom_ || !has_path_) return;

        size_t N = path_.poses.size();
        if (N < 2) return;

        // 2. FIXED: Find Closest Point with Local Minimum Break
        double min_dist = 1e9;
        size_t idx_closest = 0;
        for (size_t i = 0; i < N; ++i) {
            double dx = path_.poses[i].pose.position.x - est_.x;
            double dy = path_.poses[i].pose.position.y - est_.y;
            double d = std::hypot(dx, dy);
            
            if (d < min_dist) { 
                min_dist = d;
                idx_closest = i;
            } else if (d > min_dist + 2.0) {
                // We are moving away from the closest point, break to avoid hairpin exit
                break;
            }
        }

        // 3. FIXED: Lookahead Logic with Forward-Enforcement
        double L_ld = std::max(2.0, this->get_parameter("L_min").as_double() + 
                      this->get_parameter("k_pure").as_double() * std::abs(est_.vx));
        
        size_t idx_ld = idx_closest;
        double local_x = 0.0, local_y = 0.0;
        double gx = 0.0, gy = 0.0;

        while (idx_ld < N - 1) {
            gx = path_.poses[idx_ld].pose.position.x;
            gy = path_.poses[idx_ld].pose.position.y;
            
            double dx_global = gx - est_.x;
            double dy_global = gy - est_.y;
            double dist = std::hypot(dx_global, dy_global);

            local_x = std::cos(est_.psi) * dx_global + std::sin(est_.psi) * dy_global;
            local_y = -std::sin(est_.psi) * dx_global + std::cos(est_.psi) * dy_global;

            // Must reach distance AND point must be in front of the car
            if (dist >= L_ld && local_x > 0.1) { 
                break; 
            }
            idx_ld++;
        }

        if (local_x <= 0.0) {
            local_y = 0.0; // Force straight to avoid spinning out if confused
        }

        // 4. Pure Pursuit Steering
        double dist_sq = local_x*local_x + local_y*local_y;
        double curvature = 0.0;
        if (dist_sq > 0.001) {
            curvature = (2.0 * local_y) / dist_sq;
        }
        
        double delta = std::atan(curvature * this->get_parameter("L_base").as_double());
        double delta_max = this->get_parameter("delta_max").as_double();
        delta = std::clamp(delta, -delta_max, delta_max);

// 5. Publish
        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.header.stamp = this->now();
        drive_msg.drive.steering_angle = delta;
        drive_msg.drive.speed = std::max(1.5, this->get_parameter("target_velocity").as_double());
        
        drive_msg.drive.acceleration = 1.0; 
        
        drive_pub_->publish(drive_msg);

        // 6. Debug Marker
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map"; 
        marker.header.stamp = this->now();
        marker.ns = "lookahead";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = gx;
        marker.pose.position.y = gy;
        marker.scale.x = 0.5; marker.scale.y = 0.5; marker.scale.z = 0.5;
        marker.color.a = 1.0; marker.color.r = 1.0;
        vis_pub_->publish(marker);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PurePursuitNode>());
    rclcpp::shutdown();
    return 0;
}