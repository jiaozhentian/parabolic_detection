import cv2
import time
from __init__ import logger
import numpy as np
import argparse
from src.motion_detector_knn import MotionDetector
from src.byte_tracker import BYTETracker
from utils.visualize import plot_tracking

def make_parser():
    parser = argparse.ArgumentParser(description='Bytes Track demo')
    parser.add_argument("-f", "--fps", type=int, default=30, required=False, help="FPS of the video")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks, usually as same with FPS")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )

    return parser

VIDEO = "vtest.avi"

def track_main(tracker, detection_results, frame_id, image_height, image_width, test_size):
    '''
    main function for tracking
    :param args: the input arguments, mainly about track_thresh, track_buffer, match_thresh
    :param detection_results: the detection bounds results, a list of [x1, y1, x2, y2, score]
    :param frame_id: the current frame id
    :param image_height: the height of the image
    :param image_width: the width of the image
    :param test_size: the size of the inference model
    '''
    online_targets = tracker.update(detection_results, [image_height, image_width], test_size)
    online_tlwhs = []
    online_ids = []
    online_scores = []
    results = []

    for target in online_targets:
        tlwh = target.tlwh
        tid = target.track_id
        vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
        if tlwh[2] * tlwh[3] > args.min_box_area or vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(target.score)
            # save results
            results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{target.score:.2f},-1,-1,-1\n"
                    )

    return online_tlwhs, online_ids

if __name__ == "__main__":
    args = make_parser().parse_args(args=[])
    # traker have to be initialized out of the track_main function, cause the trak is occurred in BYTETracker.update()
    tracker = BYTETracker(args)
    
    cap = cv2.VideoCapture('./data/' + VIDEO)
    motion_detector = MotionDetector()
    ret, frame = cap.read()
    image_height = frame.shape[0]
    image_width = frame.shape[1]
    video_writer = cv2.VideoWriter('./results/' + "knn_" + VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), 30, (image_width, image_height))
    # test_size is a tuple of (width, height), which is the size of the output size of the model for inference, here is video size
    test_size = (image_height, image_width)
    frame_id = 0
    frame_pre = frame
    while ret:
        # get the contours of the motion
        start_time = time.time()
        result = motion_detector.detect_motion(frame)
        # bytetrack requires the input inference results are (left, top, right, bottom, score) format
        # conver the contours to (left, top, right, bottom, score) format
        detection_results = []
        for contour in result:
            x, y, w, h = cv2.boundingRect(contour)
            detection_results.append([float(x), float(y), float(x+w), float(y+h), 1])
        online_im = None
        if len(detection_results):
            detection_results = np.array(detection_results)
            # track the objects
            online_tlwhs, online_ids = track_main(tracker, detection_results, frame_id, image_height, image_width, test_size)
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            # Herer is the main function about byte track
            online_im = plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id, fps=fps)
        frame_id += 1

        ret, frame = cap.read()
        if ret:
            frame_pre = frame
            if online_im is not None:
                video_writer.write(online_im)
                cv2.imshow('online_im', online_im)
            else:
                video_writer.write(frame)
                cv2.imshow('online_im', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
