from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. State Estimator (Fixed Executable Name & Remappings)
        Node(
            package='my_state_estimator',  
            executable='state_estimator_node',   
            name='state_estimator',
            remappings=[
                ('/imu/data', '/sbg_imu/data'),
                ('/joint_states', '/eufs/joint_states')
            ]
        ),

        # 2. EKF-SLAM (Now publishing /slam/odom natively)
        Node(
            package='my_slam_pkg',
            executable='ekf_slam_node',
            name='ekf_slam_node',
            output='screen',
            remappings=[
                ('/perception/cones', '/ground_truth/cones'), 
                ('/odometry', '/odometry/filtered'),
                ('/map', '/planning/cones')
            ]
        ),

        # 3. Path Planner
        Node(
            package='fsd_planner',
            executable='planner_node',
            name='path_planner',
            output='screen'
        ),

        # 4. Pure Pursuit Controller (Synced with SLAM)
        Node(
            package='my_controller_pkg',
            executable='pure_pursuit_node',
            name='pure_pursuit_node',
            output='screen',
            remappings=[
                ('/path', '/target_path'),
                # REMOVED: odometry remapping so the node uses its native /slam/odom topic
                ('cmd_autonomous', '/cmd') 
            ]
        )
    ])
