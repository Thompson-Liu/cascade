#include <sys/socket.h>
#include <netinet/in.h>
#include <cascade/cascade.hpp>
#include <cascade/service_client_api.hpp>
#include "contour_detection_dpl.hpp"

using namespace derecho::cascade;

#define BUFSIZE (256)
void collect_time(uint16_t udp_port, size_t num_messages, uint64_t* photo_id, uint64_t* extract_ts, 
        uint64_t* invoke_put_ts, uint64_t* close_loop_ts, uint64_t* load_model_us, 
        uint64_t* prepare_tensor_us, uint64_t* prediction_us, uint64_t* put_us) {
    struct sockaddr_in serveraddr, clientaddr;
    char buf[BUFSIZE];
    //STEP 1: start UDP channel
  	int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  	if (sockfd < 0) {
        std::cerr << "ERROR opening socket" << std::endl;
        return;
    }
  	int optval = 1;
  	setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (const void *)&optval , sizeof(int));
    //STEP 2: waiting for UDP message
    memset(&serveraddr, 0, sizeof(serveraddr));
    memset(&clientaddr, 0, sizeof(clientaddr));
    serveraddr.sin_family = AF_INET;
    serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);
    serveraddr.sin_port = htons(udp_port);
    if ( bind(sockfd, (struct sockaddr *) &serveraddr, sizeof(serveraddr)) < 0 ) {
        std::cerr << "Fail to bind udp port:" << udp_port << std::endl;
    	return;
    }
    socklen_t clientlen = sizeof(clientaddr);
    size_t cnt = 0;
    while (cnt < num_messages) {
        ssize_t nrecv = recvfrom(sockfd, buf, BUFSIZE, 0, (struct sockaddr *) &clientaddr, &clientlen);
        if (nrecv < 0) {
            std::cerr << "Fail to recv udp package." << std::endl;
            return;
        }
        CloseLoopReport *clr = reinterpret_cast<CloseLoopReport*>(buf);
        photo_id[cnt] = clr->photo_id;
        extract_ts[cnt] = clr->extract_ts;
        invoke_put_ts[cnt] = clr->invoke_put_ts;
        close_loop_ts[cnt] = clr->close_loop_ts;
        load_model_us[cnt] = clr->load_model_us;
        prepare_tensor_us[cnt] = clr->prepare_tensor_us;
        prediction_us[cnt] = clr->prediction_us;
        put_us[cnt] = clr->put_us;
        cnt ++;
    }
    //STEP 3: finish 
    close(sockfd);
}

int main(int argc, char** argv) {
    size_t num_messages = 100;
    uint16_t udp_port = 54321;

    // collect stats
    uint64_t photo_id[num_messages];
    uint64_t extract_ts[num_messages];
    uint64_t invoke_put_ts[num_messages];
    uint64_t close_loop_ts[num_messages]; // timestamp in nanoseconds
    uint64_t load_model_us[num_messages];
    uint64_t prepare_tensor_us[num_messages];
    uint64_t prediction_us[num_messages];
    uint64_t put_us[num_messages]; // "put to pers" time cost in microsecond
    std::thread cl_thread(collect_time, udp_port, num_messages, (uint64_t*)photo_id, (uint64_t*)close_loop_ts, (uint64_t*)extract_ts, 
        (uint64_t*)invoke_put_ts, (uint64_t*)load_model_us, (uint64_t*)prepare_tensor_us, (uint64_t*)prediction_us, (uint64_t*)put_us);

    // evaulate stats
    cl_thread.join();

}
