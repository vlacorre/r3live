cmake_minimum_required(VERSION 3.5)
project(r3live)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++17 -O3 -lboost_system -msse2 -msse3 -pthread -Wenum-compare") # -Wall
set(CMAKE_CXX_STANDARD 17)
add_definitions(-w)
# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()


add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

# Add OpenCV if available
# set(OpenCV_DIR "/home/zivlin/opt/opencv/build")
FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem iostreams program_options system serialization)
if(Boost_FOUND)
	INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
	LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
endif()


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3)

# find_package(catkin REQUIRED COMPONENTS
#                     roscpp
#                     std_msgs
#                     geometry_msgs
#                     nav_msgs
#                     tf
#                     cv_bridge
#                     livox_ros_driver
#             )

# find_package(Ceres REQUIRED)
find_package(PCL 1.12 REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(OpenCV 4.5.4 EXACT REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

### Find OpenMP #######
FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	#cmake only check for separate OpenMP library on AppleClang 7+
	#https://github.com/Kitware/CMake/blob/42212f7539040139ecec092547b7d58ef12a4d72/Modules/FindOpenMP.cmake#L252
	if (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang" AND (NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.0"))
		SET(OpenMP_LIBS ${OpenMP_libomp_LIBRARY})
		LIST(APPEND OpenMVS_EXTRA_LIBS ${OpenMP_LIBS})
	endif()
else()
	message("-- Can't find OpenMP. Continuing without it.")
endif()

# find_package(OpenMVS)
if(OpenMVS_FOUND)
	include_directories(${OpenMVS_INCLUDE_DIRS})
	add_definitions(${OpenMVS_DEFINITIONS})
endif()

# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

# include_directories(${catkin_INCLUDE_DIRS})

# TODO?: for ROS 2?
# rosidl_generate_interfaces($(PROJECT_NAME)
#  DEPENDENCIES geometry_msgs
# )

# generate_messages(
#  DEPENDENCIES
#  geometry_msgs
# )

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include_directories(
  ./src
  ./src/loam/include
  ./src/tools/
  ./src/rgb_map
  ./src/meshing
)


# catkin_package()

message(STATUS "===== ${PROJECT_NAME}: OpenCV library status: =====")
message(STATUS "===== OpenCV ersion: ${OpenCV_VERSION}  =====")
message(STATUS "===== OpenCV libraries: ${OpenCV_LIBS} =====")
message(STATUS "===== OpenCV include path: ${OpenCV_INCLUDE_DIRS} =====")

add_executable(r3live_LiDAR_front_end src/loam/LiDAR_front_end.cpp)
ament_target_dependencies(r3live_LiDAR_front_end
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_conversions
  cv_bridge)
target_link_libraries(r3live_LiDAR_front_end
  ${PCL_LIBRARIES})

install(TARGETS
  r3live_LiDAR_front_end
  DESTINATION lib/${PROJECT_NAME}
)

if(Ceres_FOUND)
  message(STATUS "===== Find ceres, Version ${Ceres_VERSION} =====")
  include_directories(${CERES_INCLUDE_DIRS})
  add_executable(r3live_cam_cali src/r3live_cam_cali.cpp)
  ament_target_dependencies(r3live_cam_cali
    rclcpp
    std_msgs
    geometry_msgs
    nav_msgs
    # livox_ros_driver
   tf2
   tf2_ros
   tf2_geometry_msgs
    cv_bridge)
  target_link_libraries(r3live_cam_cali
    ${OpenCV_LIBRARIES}
    ${CERES_LIBRARIES})

    install(TARGETS
      r3live_cam_cali
      DESTINATION lib/${PROJECT_NAME}
    )

  add_executable(r3live_cam_cali_create_cali_board src/r3live_cam_cali_create_cali_board.cpp)
  ament_target_dependencies(r3live_cam_cali_create_cali_board
    rclcpp
    std_msgs
    geometry_msgs
    nav_msgs
    # livox_ros_driver
   tf2
   tf2_ros
   tf2_geometry_msgs
    cv_bridge)
  target_link_libraries(r3live_cam_cali_create_cali_board
    ${OpenCV_LIBRARIES})

    install(TARGETS
      r3live_cam_cali_create_cali_board
      DESTINATION lib/${PROJECT_NAME}
    )
endif()
add_executable(test_timer src/tools/test_timer.cpp)


install(TARGETS
  test_timer
  DESTINATION lib/${PROJECT_NAME}
)
add_executable(r3live_mapping src/r3live.cpp
                src/r3live_lio.cpp
                src/loam/include/kd_tree/ikd_Tree.cpp
                src/loam/include/FOV_Checker/FOV_Checker.cpp
                src/loam/IMU_Processing.cpp
                src/rgb_map/offline_map_recorder.cpp
                # From VIO
                src/r3live_vio.cpp
                src/optical_flow/lkpyramid.cpp
                src/rgb_map/rgbmap_tracker.cpp
                src/rgb_map/image_frame.cpp
                src/rgb_map/pointcloud_rgbd.cpp
              )


install(TARGETS
  r3live_mapping
  DESTINATION lib/${PROJECT_NAME}
)
ament_target_dependencies(r3live_mapping
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  # livox_ros_driver
  pcl_conversions
  tf2
  tf2_ros
  tf2_geometry_msgs
  PCL
  cv_bridge)
target_link_libraries(r3live_mapping
  ${Boost_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SERIALIZATION_LIBRARY} # serialization
  ${OpenCV_LIBRARIES}
  ${OpenMVS_LIBRARIES}
  ${PCL_LIBRARIES})


FIND_PACKAGE(CGAL)
if(CGAL_FOUND)
	include_directories(${CGAL_INCLUDE_DIRS})
	add_definitions(${CGAL_DEFINITIONS})
	link_directories(${CGAL_LIBRARY_DIRS})
  ADD_DEFINITIONS(-D_USE_BOOST -D_USE_EIGEN -D_USE_OPENMP)
  SET(_USE_BOOST TRUE)
  SET(_USE_OPENMP TRUE)
  SET(_USE_EIGEN TRUE)
  add_executable(r3live_meshing src/r3live_reconstruct_mesh.cpp
                                src/rgb_map/image_frame.cpp
                                src/rgb_map/pointcloud_rgbd.cpp
                                # Common
                                src/meshing/MVS/Common/Common.cpp
                                src/meshing/MVS/Common/Log.cpp
                                src/meshing/MVS/Common/Timer.cpp
                                src/meshing/MVS/Common/Types.cpp
                                src/meshing/MVS/Common/Util.cpp
                                # MVS
                                src/meshing/MVS/Mesh.cpp
                                src/meshing/MVS/PointCloud.cpp
                                src/meshing/MVS/Camera.cpp
                                src/meshing/MVS/Platform.cpp
                                src/meshing/MVS/PLY.cpp
                                src/meshing/MVS/OBJ.cpp
                                src/meshing/MVS/IBFS.cpp
                                )


install(TARGETS
  r3live_meshing
  DESTINATION lib/${PROJECT_NAME}
)
ament_target_dependencies(r3live_meshing
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  # livox_ros_driver
   tf2
   tf2_ros
   tf2_geometry_msgs
  cv_bridge
  PCL)
target_link_libraries(r3live_meshing
  ${Boost_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SERIALIZATION_LIBRARY} # serialization
  ${CGAL_LIBS}
  ${OpenCV_LIBRARIES}
  ${JPEG_LIBRARIES}
  ${PCL_LIBRARIES}
  ${PCL_KDTREE_LIBRARIES}
  ${GMP_LIBRARIES})

endif()


install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY ../config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()