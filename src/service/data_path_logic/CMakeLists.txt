cmake_minimum_required(VERSION 3.10.0)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

add_library(console_printer_dpl SHARED console_printer_dpl.cpp)
target_include_directories(console_printer_dpl PRIVATE
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
)

if (HAS_MXNET_CPP AND OPENCV_CORE_FOUND)
   add_library(image_pipeline_dpl SHARED image_pipeline_dpl.cpp)
    target_include_directories(image_pipeline_dpl PRIVATE
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
        $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
    )
    target_link_libraries(image_pipeline_dpl mxnet cascade)

    add_custom_command(TARGET image_pipeline_dpl POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/download_cnn_classifier_models.sh
        ${CMAKE_CURRENT_BINARY_DIR}/download_cnn_classifier_models.sh
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/video_receive.py
        ${CMAKE_CURRENT_BINARY_DIR}/video_receive.py
    )

    # azure
    find_package(azure-storage-blobs-cpp CONFIG REQUIRED)
    # find_library(XML2_LIBRARY xml2)
    add_library(azure_upload_dpl SHARED azure_upload_dpl.cpp)
    target_include_directories(azure_upload_dpl PRIVATE
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    )
    target_link_libraries(azure_upload_dpl cascade Azure::azure-storage-blobs)
endif()


##TODO: enable cnn_classifier dpl after data path logic manager is finished.
##if (HAS_MXNET_CPP AND OPENCV_CORE_FOUND)
##    add_library(cnn_classifier_dpl SHARED cnn_classifier_dpl.cpp)
##    target_include_directories(cnn_classifier_dpl PRIVATE
##        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
##        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
##        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
##        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
##        $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
##    )
##    target_link_libraries(cnn_classifier_dpl mxnet cascade)
##
##    add_executable(cnn_classifier_client cnn_classifier_client.cpp)
##    target_include_directories(cnn_classifier_client PRIVATE
##        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
##        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
##        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
##        $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
##    )
##    target_link_libraries(cnn_classifier_client cascade mxnet ${OpenCV_LIBS})
##    add_custom_command(TARGET cnn_classifier_client POST_BUILD
##        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/cnn_classifier_cfg
##        ${CMAKE_CURRENT_BINARY_DIR}/cnn_classifier_cfg
##        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/download_cnn_classifier_models.sh
##        ${CMAKE_CURRENT_BINARY_DIR}/download_cnn_classifier_models.sh)
##
##    add_executable(cnn_classifier_perf cnn_classifier_perf.cpp)
##    target_include_directories(cnn_classifier_perf PRIVATE
##        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
##        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
##        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
##        $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
##    )
##    target_link_libraries(cnn_classifier_perf cascade mxnet ${OpenCV_LIBS})
##endif()
