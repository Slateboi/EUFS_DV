#ifndef PURE_PURSUIT_HPP
#define PURE_PURSUIT_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

struct StateEstimate {
    double x;
    double y;
    double psi; // Heading in radians
    double vx;  // Longitudinal velocity
};

struct Path {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> v_profile; // Target velocity profile
};

struct CtrlParams {
    double L_base;    // Wheelbase (m)
    double L_min;     // Minimum lookahead distance (m)
    double k_pure;    // Lookahead gain (s)
    double Kp_v;      // Proportional gain for speed control
    double Ki_v;      // Integral gain for speed control
    double delta_max; // Max steering angle (rad)
    double acc_max;   // Max acceleration (m/s^2)
    double decel_max; // Max deceleration (m/s^2) - positive value
};

class PurePursuitController {
public:
    PurePursuitController() 
        : idx_closest_(0), vel_int_(0.0) {}

    void reset() {
        idx_closest_ = 0;
        vel_int_ = 0.0;
    }

    // Computes control commands: delta (steering), acc (acceleration)
    void computeControl(const StateEstimate& est, 
                        const Path& path, 
                        const CtrlParams& ctrl, 
                        double dt, 
                        double& delta_out, 
                        double& acc_out,
                        size_t& debug_idx) 
    {
        size_t N = path.x.size();
        if (N == 0) {
            delta_out = 0.0;
            acc_out = 0.0;
            return;
        }

        // --- 1. SEARCH FOR NEAREST POINT (Tracking) ---
        // Search window: check next 50 points to handle lap wraps efficiently
        double min_dist_sq = std::numeric_limits<double>::max();
        size_t best_idx = idx_closest_;
        
        int search_window = 50; 
        for (int i = 0; i < search_window; ++i) {
            size_t curr_idx = (idx_closest_ + i) % N;
            double dx = path.x[curr_idx] - est.x;
            double dy = path.y[curr_idx] - est.y;
            double d2 = dx*dx + dy*dy;
            
            if (d2 < min_dist_sq) {
                min_dist_sq = d2;
                best_idx = curr_idx;
            }
        }
        idx_closest_ = best_idx;
        debug_idx = idx_closest_;

        // --- 2. LOOKAHEAD DISTANCE CALCULATION ---
        // Adaptive Lookahead: L = L_min + k * vx
        double L_ld = ctrl.L_min + ctrl.k_pure * est.vx;
        
        // Find the first point on the path that is at least L_ld away
        size_t idx_ld = idx_closest_;
        for (size_t i = 0; i < N; ++i) { // Safety loop limit
            size_t curr = (idx_closest_ + i) % N;
            double dx = path.x[curr] - est.x;
            double dy = path.y[curr] - est.y;
            if (std::hypot(dx, dy) >= L_ld) {
                idx_ld = curr;
                break;
            }
        }

        // --- 3. STEERING GEOMETRY (Pure Pursuit) ---
        // Transform target point to vehicle frame (implicitly via alpha calculation)
        double target_dx = path.x[idx_ld] - est.x;
        double target_dy = path.y[idx_ld] - est.y;
        
        // Angle to target relative to map frame
        double target_heading = std::atan2(target_dy, target_dx);
        
        // Alpha: Angle between vehicle heading and target vector
        double alpha = wrapToPi(target_heading - est.psi);
        
        // Pure Pursuit Control Law: delta = atan(2 * L * sin(alpha) / L_ld)
        double ld_dist = std::hypot(target_dx, target_dy);
        if (ld_dist < 0.1) ld_dist = 0.1; // Prevent division by zero

        double delta = std::atan2(2.0 * ctrl.L_base * std::sin(alpha), ld_dist);
        
        // Clamp steering
        delta_out = std::clamp(delta, -ctrl.delta_max, ctrl.delta_max);

        // --- 4. SPEED CONTROL (Longitudinal PI) ---
        double v_target = path.v_profile[idx_closest_];
        double v_err = v_target - est.vx;
        
        // Integral action with anti-windup
        vel_int_ += v_err * dt;
        vel_int_ = std::clamp(vel_int_, -5.0, 5.0); 
        
        double acc = ctrl.Kp_v * v_err + ctrl.Ki_v * vel_int_;
        
        // Actuator Constraints
        acc_out = std::clamp(acc, -ctrl.decel_max, ctrl.acc_max);
    }

private:
    size_t idx_closest_;
    double vel_int_;

    inline double wrapToPi(double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }
};

#endif // PURE_PURSUIT_HPP