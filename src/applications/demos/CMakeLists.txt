cmake_minimum_required(VERSION 3.12.4)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
project(cascade CXX)

find_library(TENSORFLOW_LIB tensorflow)

if (TENSORFLOW_LIB AND OPENCV_CORE_FOUND)
    add_library(contour_detection_dpl SHARED contour_detection_dpl.cpp)
    target_include_directories(contour_detection_dpl PRIVATE
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
        $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${TENSORFLOW_LIB_INCLUDE_DIRS}>
    )
    target_link_libraries(contour_detection_dpl cascade ${OpenCV_LIBS} ${TENSORFLOW_LIB})

    add_custom_command(TARGET contour_detection_dpl POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/video_receive.py
        ${CMAKE_CURRENT_BINARY_DIR}/video_receive.py
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/contour-detection-model.tar.gz
        ${CMAKE_CURRENT_BINARY_DIR}/contour-detection-model.tar.gz
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/extract_contour_detection_model.sh
        ${CMAKE_CURRENT_BINARY_DIR}/extract_contour_detection_model.sh
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/demo_cfg
        ${CMAKE_CURRENT_BINARY_DIR}/demo_cfg
        COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/demo_cfg/run.sh.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/demo_cfg/n0/run.sh 
        COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/demo_cfg/run.sh.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/demo_cfg/n1/run.sh 
        COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/demo_cfg/run.sh.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/demo_cfg/n2/run.sh 
        COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/demo_cfg/run.sh.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/demo_cfg/n3/run.sh 
        COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/demo_cfg/run.sh.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/demo_cfg/n4/run.sh 
        COMMENT "prepare demo aaplication configuration"
    )

    add_executable(contour_detection_perf contour_detection_perf.cpp)
    target_include_directories(contour_detection_perf PRIVATE
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
    )
    target_link_libraries(contour_detection_perf cascade ${OpenCV_LIBS})
endif()
