from __init__ import *
import cv2


class MotionDetector(object):
    def __init__(self):
        pass
    
    def detect_motion(self, prev_frame, curr_frame):
        """
        Detects motion in a frame.
        """
        # Calculate the absolute difference between the current frame and the previous frame
        frame_diff = cv2.absdiff(curr_frame, prev_frame)
        # Convert the frame difference to grayscale
        gray_frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        # Blur the grayscale frame difference
        blur_frame_diff = cv2.GaussianBlur(gray_frame_diff, (5, 5), 0)
        # Threshold the blurred frame diffe
        # Threshold the frame difference to get only motion areas
        thresh_frame_diff = cv2.threshold(blur_frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        # open the thresholded frame difference
        # thresh_frame_diff = cv2.morphologyEx(thresh_frame_diff, cv2.MORPH_OPEN, None, iterations=2)
        cv2.imshow("thresh_frame_diff", thresh_frame_diff)
        # Get the contours
        contours, hierarchy = cv2.findContours(thresh_frame_diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # return the contours
        return contours
        # return thresh_frame_diff