#include <cascade/data_path_logic_interface.hpp>
// #include <cascade/service.hpp>
#include <cascade/cascade.hpp>
#include <string>
#include <iostream>
#include <sstream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <cppflow/cppflow.h>
#include "contour_detection_dpl.hpp"

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

/* Helper function to extract photo index from object key */
void parse_key (const std::string &obj_key, CloseLoopReport &clr) {
    std::string begin_delimiter("/"), end_delimiter("_");
    // parse photo index
    size_t begin = obj_key.rfind(begin_delimiter);
    size_t end = obj_key.find(end_delimiter, begin+1);
    std::istringstream idx_ss(obj_key.substr(begin+1, end-begin-1));
    uint64_t _photo_id;
    idx_ss >> _photo_id;
    clr.photo_id = _photo_id;
    // parse photo extract timestamp
    begin = end;
    end = obj_key.find(end_delimiter, begin+1);
    std::istringstream extract_ss(obj_key.substr(begin+1, end-begin-1));
    uint64_t _extract_ts;
    extract_ss >> _extract_ts;
    clr.extract_ts = _extract_ts;
    // parse invoke put timestamp
    std::istringstream invoke_put_ss(obj_key.substr(end+1));
    uint64_t _invoke_put_ts;
    invoke_put_ss >> _invoke_put_ts;
    clr.invoke_put_ts = _invoke_put_ts;
}

// TODO: Should move into cascade config
#define MODEL_DIRECTORY              "../../model"  

class ContourDetectionTrigger: public OffCriticalDataPathObserver {
private:
    mutable std::mutex p2p_send_mutex;
#ifdef EVALUATION
    int sock_fd;
    struct sockaddr_in serveraddr;
#endif

public:
    ContourDetectionTrigger (): OffCriticalDataPathObserver()
    {
#ifdef EVALUATION
#define DPL_CONF_REPORT_TO	"CASCADE/report_to"
        uint16_t port;
        struct hostent *server;
        std::string hostname;
        std::string report_to = derecho::getConfString(DPL_CONF_REPORT_TO);
        hostname = report_to.substr(0, report_to.find(":"));
        port = (uint16_t)std::stoi(report_to.substr(report_to.find(":") + 1));
        if ( (sock_fd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) {
            std::cerr << "Faile to open socket" << std::endl;
            return;
        }
        if ( !(server = gethostbyname(hostname.c_str())) ) {
            std::cerr << "Failed to get host:" << hostname << std::endl;
            return;
        }
        memset(&serveraddr, 0, sizeof(serveraddr));
        serveraddr.sin_family = AF_INET;
        bcopy((char *)server->h_addr, (char *)&serveraddr.sin_addr.s_addr, server->h_length);
        serveraddr.sin_port = htons(port);
#endif
    }

    virtual void operator () (const std::string& key_string,
                              persistent::version_t version,
                              const mutils::ByteRepresentable* const value_ptr,
                              ICascadeContext* ctxt,
                              uint32_t worker_id) override {
#ifdef EVALUATION
        CloseLoopReport clr;
        clr.close_loop_ts = get_time();
        parse_key(key_string, clr);
        uint64_t before_load_model_ns = get_time();
#endif
        /* step 1: load the model */ 
        static thread_local cppflow::model model(MODEL_DIRECTORY);
#ifdef EVALUATION
        uint64_t after_load_model_ns = get_time();
        clr.load_model_us = (after_load_model_ns-before_load_model_ns)/1000;
#endif
        /* step 2: Load the image & convert to tensor */
        // TODO: This introduces another copy, should avoid by separating this logic into client side
        VolatileCascadeStoreWithStringKey::ObjectType *vcss_value = const_cast<VolatileCascadeStoreWithStringKey::ObjectType*>(reinterpret_cast<const VolatileCascadeStoreWithStringKey::ObjectType *>(value_ptr));
        std::vector<float> preprocess_buf = std::move(preprocess_photo(reinterpret_cast<void*>(vcss_value->blob.bytes), vcss_value->blob.size)); 
        cppflow::tensor input_tensor(preprocess_buf, {PHOTO_WIDTH,PHOTO_HEIGHT,3});
        input_tensor = cppflow::expand_dims(input_tensor, 0);
#ifdef EVALUATION
        uint64_t after_prepare_tensor_ns = get_time();
        clr.prepare_tensor_us = (after_prepare_tensor_ns-after_load_model_ns)/1000;
#endif 
        /* step 3: Predict */
        cppflow::tensor output = model({{"serving_default_conv2d_3_input:0", input_tensor}},{"StatefulPartitionedCall:0"})[0];
        // prediction < 0.35 indicates strong possibility that the image frame captures full contour of the cow
        std::string prediction = std::to_string(output.get_data<float>()[0]);
#ifdef EVALUATION
        uint64_t after_prediction_ns = get_time();
        clr.prediction_us = (after_prediction_ns-after_prepare_tensor_ns)/1000;
#endif
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
#ifdef EVALUATION
        uint64_t after_put_ns = get_time();
        clr.put_us = (after_put_ns-after_prediction_ns)/1000;
        std::cout << to_string(clr);
        size_t ns = sendto(sock_fd,(const void*)&clr,sizeof(clr),0,(const struct sockaddr*)&serveraddr,sizeof(serveraddr));
        if (ns < 0) {
            std::cerr << "Failed to report error" << std::endl;
        }
#endif
    }

    virtual ~ContourDetectionTrigger() {
#ifdef EVALUATION
        close(sock_fd);
#endif
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