#pragma once
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <cascade_farm.grpc.pb.h>

using cascade_frame::FrameInfo;
using cascade_frame::FrameService;
using cascade_frame::FrameStatus;
using cascade_frame::UploadStatusCode;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

const std::string RPC_FAILURE = "RPC failed";

FrameInfo MakeFrameInfo(const std::string &inputSource, uint32_t frameIdx)
{
    FrameInfo info;
    info.set_inputsource(inputSource);
    info.set_frameidx(frameIdx);
    return info;
}

class FrameClient
{
public:
    FrameClient(std::shared_ptr<Channel> channel)
        : stub_(FrameService::NewStub(channel)) {}

    // Assembles the client's payload, sends it and presents the response back
    // from the server.
    std::string GetFrame(const FrameInfo &request, FrameStatus *reply)
    {
        // Context for the client. It could be used to convey extra information to
        // the server and/or tweak certain RPC behaviors.
        ClientContext context;

        // The actual RPC.
        Status status = stub_->GetFrame(&context, request, reply);

        // Act upon its status.
        if (status.ok())
        {
            return reply->frame_data();
        }
        else
        {
            std::cout << status.error_code() << ": " << status.error_message() << std::endl;
            return RPC_FAILURE;
        }
    }

private:
    std::unique_ptr<FrameService::Stub> stub_;
};

std::string request_frame(std::string& source, int frame_idx)
{
    std::cout << "Client executed to request a frame" << std::endl;
    FrameClient frameGetter(grpc::CreateChannel(
        "localhost:50051", grpc::InsecureChannelCredentials()));
    FrameInfo info = MakeFrameInfo(source, frame_idx);
    FrameStatus status;
    return frameGetter.GetFrame(info, &status);
}