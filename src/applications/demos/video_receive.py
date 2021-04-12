import socket
import cv2
import pickle
import struct
import sys
import numpy as np
import cascade_py

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
            source, idx = frame_packet['header']
            image_frame = frame_packet['frame']
            print(source)
            print(idx)

            cascade_frame = image_frame.tobytes()
            ret = capi.put('VCSS', '/demo/contour_detection/'+str(idx), cascade_frame, 0, 0)
            print(ret.get_result())
        except Exception as e:
            print('exception occured: {}'.format(e))
    receive_socket.close()


if __name__ == '__main__':
    host_ip = ''
    port = 8081
    receive_frames_from_server(host_ip, port)
