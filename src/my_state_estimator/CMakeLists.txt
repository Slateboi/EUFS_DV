cmake_minimum_required(VERSION 3.5)
project(my_state_estimator)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(state_estimator_node src/State_Estimator.cpp)
target_include_directories(state_estimator_node PRIVATE ${Eigen3_INCLUDE_DIRS})
ament_target_dependencies(state_estimator_node rclcpp sensor_msgs std_msgs nav_msgs tf2)

install(TARGETS state_estimator_node DESTINATION lib/${PROJECT_NAME})
ament_package()
