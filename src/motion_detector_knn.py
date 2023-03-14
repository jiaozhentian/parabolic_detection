import cv2
from matplotlib.pyplot import contour
import numpy as np

class MotionDetector(object):
    def __init__(self):
        self.video_skip = True
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.color_m = (255, 0, 0)
        self.knn = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=500, detectShadows=True)

    def detect_motion(self, frame):
        # frame = cv2.resize(frame, (500, 500), interpolation=cv2.INTER_CUBIC)
        frame_motion = frame.copy()
        fgmask = self.knn.apply(frame_motion)
        draw1 = cv2.threshold(fgmask, 50, 255, cv2.THRESH_BINARY)[1]
        draw1 = cv2.morphologyEx(draw1, cv2.MORPH_OPEN, self.kernel, iterations=1)

        contours_m, hierarchy_m = cv2.findContours(draw1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # for c in contours_m:
        #     if cv2.contourArea(c) < 300:
        #         continue
        #     (x, y, w, h) = cv2.boundingRect(c)
        #     cv2.rectangle(frame_motion, (x, y), (x + w, y + h), self.color_m, 2)
        # cv2.imshow("Motion", frame_motion)
        # return draw1
        # filter the contours if the area is too small
        contours_m = [c for c in contours_m if cv2.contourArea(c) > 300]
        return contours_m