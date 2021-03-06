import socket
import cv2
import pickle
import struct
import sys
import numpy as np
import cascade_py
import time

capi = cascade_py.ServiceClientAPI()

def receive_frames_from_server(host_ip, port):
    receive_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    receive_socket.bind((host_ip, port))
    print("[SOCKET] bind complete")

    receive_socket.listen(10)
    print("[SOCKET] listening ")
    conn, addr = receive_socket.accept()

    data = b""
    payload_size = struct.calcsize("Q")
    print("payload_size: {}".format(payload_size))
    while True:
        try:
            while len(data) < payload_size:
                # use 4b buffer
                data += conn.recv(4096)

            print("Done Recv: {}".format(len(data)))
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]
            while len(data) < msg_size:
                data += conn.recv(4096)
            frame_data = data[:msg_size]
            data = data[msg_size:]
            frame_packet = pickle.loads(frame_data)
            ts,idx = frame_packet['header']
            image_frame = frame_packet['frame']

            cascade_frame = image_frame.tobytes()
            # /demo/contour_detection/<idx>_<extract_ts_ns>_<put_ts_ns>
            extract_ts = "{:.0f}".format(ts*(10**9))
            invoke_put_ts = "{:.0f}".format(time.time()*(10**9))
            frame_id = '/demo/contour_detection/'+str(idx)+'_'+extract_ts+'_'+invoke_put_ts
            print(frame_id)
            ret = capi.put('VCSS', frame_id, cascade_frame, 0, 0)
            print(ret.get_result())
        except Exception as e:
            print('exception occured: {}'.format(e))
    receive_socket.close()


if __name__ == '__main__':
    host_ip = ''
    port = 8081
    receive_frames_from_server(host_ip, port)
