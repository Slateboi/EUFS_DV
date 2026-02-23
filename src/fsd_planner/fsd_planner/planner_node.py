import rclpy
from rclpy.node import Node
import numpy as np
import math
import sys

from eufs_msgs.msg import ConeArray
from nav_msgs.msg import Path, Odometry # FIX: Imported Odometry
from geometry_msgs.msg import PoseStamped

# --- IMPORT FSD LIBRARY ---
lib_path = '/home/mayuresh/eufs_ws/src/fsd_planner/fsd_planner/lib/ft_fsd_path_planning'
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

try:
    from fsd_path_planning.full_pipeline.full_pipeline import PathPlanner
    from fsd_path_planning.utils.mission_types import MissionTypes
    from fsd_path_planning.utils.cone_types import ConeTypes
    print("SUCCESS: PathPlanner Global Pipeline Imported!")
except ImportError:
    print(f"ERROR: Could not find library at {lib_path}")
    PathPlanner = None

class FSDPlannerNode(Node):
    def __init__(self):
        super().__init__('fsd_planner_node')

        self.planner = PathPlanner(MissionTypes.trackdrive)

        # FIX 1: Use default Reliable QoS (10) to match the SLAM publishers
        self.sub_cones = self.create_subscription(
            ConeArray, 
            '/planning/cones', 
            self.cone_callback, 
            10)
            
        # FIX 2: Subscribe to SLAM Odometry, NOT the raw car state!
        self.sub_state = self.create_subscription(
            Odometry, 
            '/slam/odom', 
            self.state_callback, 
            10)
        
        self.pub_path = self.create_publisher(Path, '/target_path', 10)

        self.car_x, self.car_y, self.car_yaw = 0.0, 0.0, 0.0
        self.got_state = False
        self.get_logger().info("Planner: FSD SLAM-Based Path Provider Online. Listening to /slam/odom")

    def state_callback(self, msg):
        # Parses the nav_msgs/Odometry message from SLAM
        self.car_x = msg.pose.pose.position.x
        self.car_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.car_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.got_state = True

    def cone_callback(self, msg):
        if not self.got_state: return 

        cones_input = [np.empty((0, 2))] * 5 
        def process_cones(ros_cones):
            if not ros_cones: return np.empty((0, 2))
            return np.array([[c.x, c.y] for c in ros_cones])

        cones_input[ConeTypes.BLUE] = process_cones(msg.blue_cones)
        cones_input[ConeTypes.YELLOW] = process_cones(msg.yellow_cones)
        cones_input[ConeTypes.ORANGE_BIG] = process_cones(msg.big_orange_cones)

        if len(cones_input[ConeTypes.BLUE]) == 0 and len(cones_input[ConeTypes.YELLOW]) == 0:
            return

        # Planner now correctly calculates path relative to SLAM's corrected pose
        vehicle_pos = np.array([self.car_x, self.car_y])
        vehicle_dir = np.array([np.cos(self.car_yaw), np.sin(self.car_yaw)])

        try:
            full_path = self.planner.calculate_path_in_global_frame(
                cones_input, vehicle_pos, vehicle_dir
            )
            if full_path is not None and len(full_path) > 0:
                path_xy = full_path[:, 1:3]
                if not (np.any(np.isnan(path_xy)) or np.any(np.isinf(path_xy))):
                    self.publish_path(path_xy)
        except Exception as e:
            self.get_logger().error(f"Planning Pipeline Error: {e}")

    def publish_path(self, np_path):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        for pt in np_path:
            pose = PoseStamped()
            pose.pose.position.x, pose.pose.position.y = float(pt[0]), float(pt[1])
            path_msg.poses.append(pose)
        self.pub_path.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FSDPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
