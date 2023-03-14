import cv2
import numpy as np
import imageio

def video2gif(video, save_path):
    video_capture = cv2.VideoCapture(video)
    success, frame = video_capture.read()
    frames = []
    count = 0
    while success:
        count += 1
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if count % 2 == 0:
            frames.append(frame)
            count = 0
        success, frame = video_capture.read()
    imageio.mimsave(save_path, frames[:30], 'GIF', duration=2/30)

if __name__ == '__main__':
    video2gif('./results/knn_vtest.avi', './results/knn_vtest.gif')