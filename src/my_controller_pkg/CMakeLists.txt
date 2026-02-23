cmake_minimum_required(VERSION 3.8)
project(my_controller_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find all required ROS 2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED) # <--- This fixes your error
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

add_executable(pure_pursuit_node src/pure_pursuit_controller.cpp)
ament_target_dependencies(pure_pursuit_node
  rclcpp
  nav_msgs
  ackermann_msgs
  geometry_msgs
  visualization_msgs
  tf2
  tf2_geometry_msgs
)

install(TARGETS pure_pursuit_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

ament_package()
