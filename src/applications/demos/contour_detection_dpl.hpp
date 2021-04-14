#pragma once
#include <vector>
#include <string>
#include <sstream>
#include <opencv2/opencv.hpp>

#define PHOTO_HEIGHT                (240)
#define PHOTO_WIDTH                 (352)

std::vector<float> preprocess_photo(void* in_buf, size_t in_size) {
    std::vector<unsigned char> decode_buf(in_size);
    std::memcpy(static_cast<void*>(decode_buf.data()),static_cast<const void*>(in_buf),in_size);
    cv::Mat mat = cv::imdecode(decode_buf, cv::IMREAD_COLOR);
    std::vector<float> array(PHOTO_HEIGHT*PHOTO_WIDTH*3);
    for (int i=0; i<PHOTO_HEIGHT; i++) {        // height
        for (int j=0; j<PHOTO_WIDTH; j++) {     // width
            for (int c=0; c<3; c++) {           // channels BGR->RGB
                // normalize to tensorflow float between [0,1]
                array[(i * PHOTO_WIDTH + j) * 3 + c] = static_cast<float>(mat.data[(i * PHOTO_WIDTH + j) * 3 + (2 - c)] / 255.f);
            }
        }
    }
    return array;
}

typedef struct __attribute__ ((packed)) {
    uint64_t    photo_id;
    uint64_t    extract_ts;
    uint64_t    invoke_put_ts;
    uint64_t    close_loop_ts;
    uint64_t    load_model_us;
    uint64_t    prepare_tensor_us;
    uint64_t    prediction_us;
    uint64_t    put_us;
} CloseLoopReport;

inline std::string to_string (CloseLoopReport clr) {
    std::ostringstream out;
    out << "CloseLoopReport: \n";
    out << "\t-[photo_id: " << std::to_string(clr.photo_id) << "]\n";
    out << "\t-[extract_ts: " << std::to_string(clr.extract_ts) << "]\n";
    out << "\t-[invoke_put_ts: " << std::to_string(clr.invoke_put_ts) << "]\n";
    out << "\t-[close_loop_ts: " << std::to_string(clr.close_loop_ts) << "]\n";
    out << "\t-[load_model_us: " << std::to_string(clr.load_model_us) << "]\n";
    out << "\t-[prepare_tensor_us: " << std::to_string(clr.prepare_tensor_us) << "]\n";
    out << "\t-[prediction_us: " << std::to_string(clr.prediction_us) << "]\n";
    out << "\t-[put_us: " << std::to_string(clr.put_us) << "]\n";
    return out.str();
}

