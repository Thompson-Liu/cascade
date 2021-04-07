cmake_minimum_required(VERSION 3.10.0)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf MODULE REQUIRED)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)
message(STATUS "Using protobuf ${protobuf_VERSION}")

set(PROTO_PATH "${CMAKE_CURRENT_SOURCE_DIR}/protos")
set(CASCADE_FARM_PROTO "${PROTO_PATH}/cascade_farm.proto")
set(GENERATED_PROTOBUF_PATH "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY ${GENERATED_PROTOBUF_PATH})

set(CASCADE_FARM_PB_CPP_FILE "${GENERATED_PROTOBUF_PATH}/cascade_farm.pb.cc")
set(CASCADE_FARM_PB_H_FILE "${GENERATED_PROTOBUF_PATH}/cascade_farm.pb.h")
set(CASCADE_FARM_GRPC_PB_CPP_FILE "${GENERATED_PROTOBUF_PATH}/cascade_farm.grpc.pb.cc")
set(CASCADE_FARM_GRPC_PB_H_FILE "${GENERATED_PROTOBUF_PATH}/cascade_farm.grpc.pb.h")
add_custom_command(
    OUTPUT "${CASCADE_FARM_PB_H_FILE}"
           "${CASCADE_FARM_PB_CPP_FILE}"
           "${CASCADE_FARM_GRPC_PB_H_FILE}"
           "${CASCADE_FARM_GRPC_PB_CPP_FILE}"
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    ARGS "--proto_path=${PROTO_PATH}"
         "--cpp_out=${GENERATED_PROTOBUF_PATH}"
         "${CASCADE_FARM_PROTO}"
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    ARGS "--proto_path=${PROTO_PATH}"
         "--grpc_out=${GENERATED_PROTOBUF_PATH}"
         "--plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}"
         "${CASCADE_FARM_PROTO}"
    DEPENDS "${CASCADE_FARM_PROTO}"
    )
set(CASCADE_FARM_PROTO_HDRS ${CASCADE_FARM_PB_H_FILE} ${CASCADE_FARM_GRPC_PB_H_FILE})
set(CASCADE_FARM_PROTO_SRCS ${CASCADE_FARM_PB_CPP_FILE} ${CASCADE_FARM_GRPC_PB_CPP_FILE})

if (HAS_MXNET_CPP AND OPENCV_CORE_FOUND)
    add_library(image_classifier_dpl SHARED image_classifier_dpl.cpp ${CASCADE_FARM_PROTO_SRCS} ${CASCADE_FARM_PROTO_HDRS})
    target_include_directories(image_classifier_dpl PRIVATE
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
        $<BUILD_INTERFACE:${GENERATED_PROTOBUF_PATH}>
        $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
    )
    target_link_libraries(image_classifier_dpl protobuf grpc++ cascade mxnet pthread)

    add_custom_command(TARGET image_classifier_dpl POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/image_pipeline_cfg
        ${CMAKE_CURRENT_BINARY_DIR}/image_pipeline_cfg
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/download_cnn_classifier_models.sh
        ${CMAKE_CURRENT_BINARY_DIR}/download_cnn_classifier_models.sh
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/video_capture.py
        ${CMAKE_CURRENT_BINARY_DIR}/video_capture.py
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/video_receive.py
        ${CMAKE_CURRENT_BINARY_DIR}/video_receive.py
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/test_videos
        ${CMAKE_CURRENT_BINARY_DIR}/test_videos
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/video_clip_store
        ${CMAKE_CURRENT_BINARY_DIR}/video_clip_store)
endif()

    
