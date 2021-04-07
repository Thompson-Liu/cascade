#pragma once
#include <iostream>
#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cascade/cascade.hpp>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <string>

#define PHOTO_HEIGHT    (224)
#define PHOTO_WIDTH     (224)
#define PHOTO_OUTPUT_BUFFER_SIZE    (PHOTO_HEIGHT*PHOTO_WIDTH*3*sizeof(mx_float))

using namespace derecho::cascade;

inline int preprocess_photo(std::string in_buf, void* out_buf) {
    dbg_default_trace("start to preprocess the data");
    std::vector<unsigned char> decode_buf(in_buf.begin(), in_buf.end());
    cv::Mat mat = cv::imdecode(decode_buf, cv::IMREAD_COLOR);
    dbg_default_trace("matrix rows: " + std::to_string(mat.rows));
    dbg_default_trace("matrix columns: " + std::to_string(mat.cols));
    std::vector<mx_float> array;
    cv::resize(mat, mat, cv::Size(256,256));
    dbg_default_trace("matrix rows after resizing: " + std::to_string(mat.rows));
    dbg_default_trace("matrix columns after resizing: " + std::to_string(mat.cols));
    for (int c=0; c<3; c++) {           // channels GBR->RGB
        for (int i=0; i<PHOTO_HEIGHT; i++) {     // height
            for (int j=0; j<PHOTO_WIDTH; j++) { // width
                int _i = i+16;
                int _j = j+16;
                array.push_back(static_cast<mx_float>(mat.data[(_i * 256 + _j) * 3 + (2 - c)]) / 256);
            }
        }
    }

    std::memcpy(out_buf,array.data(),PHOTO_OUTPUT_BUFFER_SIZE);

    return 0;
}

typedef struct __attribute__ ((packed)) {
    uint64_t    photo_id;
    char        data[PHOTO_OUTPUT_BUFFER_SIZE];
} FrameData;

typedef struct __attribute__ ((packed)) {
    uint64_t    photo_id;
} CloseLoopReport;