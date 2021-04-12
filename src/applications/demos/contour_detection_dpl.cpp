#include <cascade/data_path_logic_interface.hpp>
#include <cascade/service.hpp>
#include <cascade/cascade.hpp>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cppflow/cppflow.h>

namespace derecho{
namespace cascade{

#define MY_PREFIX   "/demo/contour_detection"
#define MY_UUID     "56d26bf3-e61d-433d-88a9-6c3a0736046a"
#define MY_DESC     "DLL DPL that detects if the image frame captures the contour of the cow"

std::unordered_set<std::string> list_prefixes() {
    return {MY_PREFIX};
}

std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}

void initialize(ICascadeContext* ctxt) {
    // load dfg
    // std::ifstream input_conf(DFG_CONFIG);
    // json dfg_conf = json::parse(input_conf); 
    // dfg_descriptor = DFGDescriptor(dfg_conf);
    // dfg_descriptor.dump();
}

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

// TODO: Should move into cascade config
#define MODEL_DIRECTORY              "../../model"  

class ContourDetectionTrigger: public OffCriticalDataPathObserver {
private:
    std::mutex p2p_send_mutex;
public:
    virtual void operator () (const std::string& key_string,
                              persistent::version_t version,
                              const mutils::ByteRepresentable* const value_ptr,
                              ICascadeContext* ctxt,
                              uint32_t worker_id) override {
        /* step 1: load the model */ 
        static thread_local cppflow::model model(MODEL_DIRECTORY);
        /* step 2: Load the image & convert to tensor */
        // TODO: This introduces another copy, should avoid by separating this logic into client side
        VolatileCascadeStoreWithStringKey::ObjectType *vcss_value = const_cast<VolatileCascadeStoreWithStringKey::ObjectType*>(reinterpret_cast<const VolatileCascadeStoreWithStringKey::ObjectType *>(value_ptr));
        std::vector<float> preprocess_buf = std::move(preprocess_photo(reinterpret_cast<void*>(vcss_value->blob.bytes), vcss_value->blob.size)); 
        cppflow::tensor input_tensor(preprocess_buf, {PHOTO_WIDTH,PHOTO_HEIGHT,3});
        input_tensor = cppflow::expand_dims(input_tensor, 0);
        /* step 3: Predict */
        cppflow::tensor output = model({{"serving_default_conv2d_3_input:0", input_tensor}},{"StatefulPartitionedCall:0"})[0];
        // prediction < 0.35 indicates strong possibility that the image frame captures full contour of the cow
        std::string prediction = std::to_string(output.get_data<float>()[0]);
        /* step 4: Send to Cascade */
        std::string prediction_key = key_string + "/prediction";
        PersistentCascadeStoreWithStringKey::ObjectType obj(prediction_key,prediction.c_str(),prediction.size());
        std::lock_guard<std::mutex> lock(p2p_send_mutex);
        auto* typed_ctxt = dynamic_cast<CascadeContext<VolatileCascadeStoreWithStringKey,PersistentCascadeStoreWithStringKey,TriggerCascadeNoStoreWithStringKey>*>(ctxt);
        auto result = typed_ctxt->get_service_client_ref().template put<PersistentCascadeStoreWithStringKey>(obj);
        for (auto& reply_future:result.get()) {
            auto reply = reply_future.second.get();
            dbg_default_debug("node({}) replied with version:({:x},{}us)",reply_future.first,std::get<0>(reply),std::get<1>(reply));
        }
    }
};

void register_triggers(ICascadeContext* ctxt) {
    // Please make sure the CascadeContext type matches the CascadeService type, which is defined in server.cpp if you
    // use the default cascade service binary.
    auto* typed_ctxt = dynamic_cast<CascadeContext<VolatileCascadeStoreWithStringKey,PersistentCascadeStoreWithStringKey,TriggerCascadeNoStoreWithStringKey>*>(ctxt);
    typed_ctxt->register_prefixes({MY_PREFIX},MY_UUID,std::make_shared<ContourDetectionTrigger>());
}

void unregister_triggers(ICascadeContext* ctxt) {
    auto* typed_ctxt = dynamic_cast<CascadeContext<VolatileCascadeStoreWithStringKey,PersistentCascadeStoreWithStringKey,TriggerCascadeNoStoreWithStringKey>*>(ctxt);
    typed_ctxt->unregister_prefixes({MY_PREFIX},MY_UUID);
}

void release(ICascadeContext* ctxt) {
    // nothing to release
    return;
}

} // namespace cascade
} // namespace derecho