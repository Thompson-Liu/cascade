import grpc
import cascade_farm_pb2
import cascade_farm_pb2_grpc
import cv2
import os
from concurrent import futures
import logging
import time
import socket
import pickle
import struct
import numpy as np
import sys
import multiprocessing as mp


def extract(input_source, q_extract_sample, q_extract_store):
    vidcap = cv2.VideoCapture(input_source)
    source = os.path.basename(input_source).split('.')[0]

    print('Extracting frames..\n')
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('total number of frames: {}'.format(length))
    print('video frame properties: width {} * height {}'.format(width, height))

    frame_idx = 0
    while True:
        success, frame = vidcap.read()
        print('processing frame idx: {}'.format(frame_idx))
        if success:
            q_extract_sample.put((source, frame_idx, frame))
            q_extract_store.put((source, frame_idx, frame))
            frame_idx += 1
        else:
            q_extract_sample.put(('', -1, None))
            q_extract_store.put(('', -1, None))
            break
    vidcap.release()


def sample_send(rate, resolution, port, q_in):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.connect(('localhost', port))
    while True:
        source, idx, frame = q_in.get()
        if idx == -1:
            server_socket.close()
            return

        if idx % rate == 0:
            frame = cv2.resize(frame, resolution)
            success_encode, buf = cv2.imencode('.jpg', frame)
            frame_packet = {'header': (source,idx), 'frame': buf}
            print(frame_packet)
            data = pickle.dumps(frame_packet)
            server_socket.sendall(struct.pack("Q", len(data)) + data)
            print('compressed and sent frame idx: {}'.format(idx))
            time.sleep(1)


def store(input, output, q_in):
    output_dir = os.path.join(
        output, os.path.basename(input).split('.')[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    while True:
        source, idx, frame = q_in.get()
        if idx == -1:
            return
        print('storing frame idx: {}'.format(idx))
        ret = cv2.imwrite(os.path.join(output_dir, str(idx)+'.jpg'), frame)


class FrameServicer(cascade_farm_pb2_grpc.FrameServiceServicer):
    def GetFrame(self, request, context):
        source = request.inputSource
        idx = request.frameIdx
        for root, _, files in os.walk('/home/ml2579/workspace/cascade_fork/cascade/build/src/service/image_pipeline/video_clip_store'):
            if root == os.path.join('/home/ml2579/workspace/cascade_fork/cascade/build/src/service/image_pipeline/video_clip_store', source) and (str(idx) + '.jpg') in files:
                img_path = os.path.join(root, str(idx) + '.jpg')
                return cascade_farm_pb2.FrameStatus(code=cascade_farm_pb2.UploadStatusCode.Ok, file_path=img_path)
        print("Requested frame not found!")
        return cascade_farm_pb2.FrameStatus(code=cascade_farm_pb2.UploadStatusCode.NotFound, content=b'')


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    cascade_farm_pb2_grpc.add_FrameServiceServicer_to_server(FrameServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


def main(port, input_source, output, sample_fps, compressed_width, compressed_height):
    pool = mp.Pool()
    manager = mp.Manager()

    # created managed queues
    q_extract_sample = manager.Queue()
    q_extract_store = manager.Queue()

    # launch workers, passing them the queues they need
    results_extract = pool.apply_async(extract, (input_source, q_extract_sample, q_extract_store))
    results_sample = pool.apply_async(sample_send, (sample_fps, (compressed_width, compressed_height), port, q_extract_sample))
    results_store = pool.apply_async(store, (input_source, output, q_extract_store))
    grpc_frame_service = pool.apply_async(serve)
    pool.close()
    pool.join()


# python frame_server.py <port> <input_source> <output_source> <sampling_rate> <compressed_width> <compressed_height>
if __name__ == '__main__':
    mp.set_start_method('spawn')
    print('start to run')
    args = sys.argv
    main(int(args[1]), args[2], args[3], int(args[4]), int(args[5]), int(args[6]))
