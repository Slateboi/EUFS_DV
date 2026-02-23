cmake_minimum_required(VERSION 3.8)
project(my_slam_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(eufs_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
# Add these three lines to the existing find_package section
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)

# Add the SLAM executable
add_executable(ekf_slam_node src/Slam.cpp)

# Link everything
target_include_directories(ekf_slam_node PRIVATE ${EIGEN3_INCLUDE_DIRS})
ament_target_dependencies(ekf_slam_node 
  rclcpp 
  sensor_msgs 
  std_msgs 
  nav_msgs 
  eufs_msgs
  tf2
  tf2_ros
  geometry_msgs
)

# Install
install(TARGETS
  ekf_slam_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
