#include <cascade/service.hpp>
#include <cascade/cascade.hpp>
#include <cascade/data_path_logic_interface.hpp>
#include <iostream>
#include <string>
#include <azure/storage/blobs.hpp>
#include <azure/core.hpp>

namespace derecho{
namespace cascade{

#define MY_PREFIX   "/image_pipeline/azure_upload"
#define MY_UUID     "48e60f7c-1000-11eb-8755-0242ac110002"
#define MY_DESC     "DLL DPL that uploads image frames to azure blob storage"

/** DFG Descriptor, will be initialized during dpl initialization **/
DFGDescriptor dfg_descriptor; 

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
    std::ifstream input_conf(DFG_CONFIG);
    json dfg_conf = json::parse(input_conf); 
    dfg_descriptor = DFGDescriptor(dfg_conf);
    dfg_descriptor.dump();
}

class ImageFrame: public Blob {
public:
    std::string key;
    ImageFrame(const std::string& k, const Blob& other): Blob(other), key(k) {}
};

#define CONNECTION_STRING   "DefaultEndpointsProtocol=https;AccountName=dairydatafiles;AccountKey=SSLgbtN3wgc3q+Fm2SxrWGZlyE1zE1q59MJzmcB1aywWTkWNZn7/q3bYychqaYBg85/LcpaW4MblJCLusBhmSw==;EndpointSuffix=core.windows.net"
#define CONTAINER           "video-frames"

using namespace Azure::Storage::Blobs;

class AzureUploadTrigger: public OffCriticalDataPathObserver {
public:
    virtual void operator () (const std::string& key_string,
                              persistent::version_t version,
                              const mutils::ByteRepresentable* const value_ptr,
                              ICascadeContext* ctxt,
                              uint32_t worker_id) override {
        std::cout << "[azure upload ocdpo]: I(" << worker_id << ") received an object with key=" << key_string << std::endl;
        std::string blob_name = "default_frame.jpg";
        std::size_t pos = key_string.rfind("/");
        if (pos != std::string::npos) {
            blob_name = key_string.substr(pos+1);
        }
        VolatileCascadeStoreWithStringKey::ObjectType *vcss_value = const_cast<VolatileCascadeStoreWithStringKey::ObjectType*>(reinterpret_cast<const VolatileCascadeStoreWithStringKey::ObjectType *>(value_ptr));
        try {
            auto containerClient = BlobContainerClient::CreateFromConnectionString(CONNECTION_STRING, CONTAINER);
            BlockBlobClient blobClient = containerClient.GetBlockBlobClient(blob_name);
            blobClient.UploadFrom(reinterpret_cast<const uint8_t*>(vcss_value->blob.bytes), vcss_value->blob.size);
            std::cout << "uploaded " << std::to_string(vcss_value->blob.size) << " bytes to azure storage account" << std::endl;
        }
        catch (const Azure::Core::RequestFailedException& e) {
            std::cout << e.what() << std::endl;
            return;
        }
        std::cout << "successfully uploaded" << std::endl;
    }
};

void register_triggers(ICascadeContext* ctxt) {
    // Please make sure the CascadeContext type matches the CascadeService type, which is defined in server.cpp if you
    // use the default cascade service binary.
    auto* typed_ctxt = dynamic_cast<CascadeContext<VolatileCascadeStoreWithStringKey,PersistentCascadeStoreWithStringKey>*>(ctxt);
    typed_ctxt->register_prefixes({MY_PREFIX},MY_UUID,std::make_shared<AzureUploadTrigger>());
}

void unregister_triggers(ICascadeContext* ctxt) {
    auto* typed_ctxt = dynamic_cast<CascadeContext<VolatileCascadeStoreWithStringKey,PersistentCascadeStoreWithStringKey>*>(ctxt);
    typed_ctxt->unregister_prefixes({MY_PREFIX},MY_UUID);
}

void release(ICascadeContext* ctxt) {
    // nothing to release
    return;
}

} // namespace cascade
} // namespace derecho