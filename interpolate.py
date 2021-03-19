'''
Code adapted from:
1) https://stackoverflow.com/questions/13685771/opencv-python-calcopticalflowfarneback
2) https://theailearner.com/2018/11/15/image-interpolation-using-opencv-python/
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image, ImageDraw


def linear_interpolate(images, interpolate_rate=5):
    '''
    Create a video interpolating between the frames by taking a linear
    weight of each pair of neighbor frames
    '''
    assert len(images) > 1
    iplated = []
    prev = images[0]
    for frame in images[1:]:
        for i in range(1, interpolate_rate+1):
            weight = float(i) / interpolate_rate
            mid_frame = cv2.addWeighted(prev, 1-weight, frame, weight, 0)
            iplated.append(Image.fromarray(mid_frame))
        prev = frame
    return iplated


def flow_interpolate(images, interpolate_rate=5):
    # Calculates dense optical flow by Farneback method
    prev = images[0]
    iplated = []
    for frame in images[1:]:
        
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY) 
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
        flow = cv2.calcOpticalFlowFarneback(prev_gray,gray, None, 0.5, 3, 25, 3, 7, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN) 

        viz_flow(frame,flow)
        h, w = flow.shape[:2]

        flow = -flow
        flow[:, :, 0] += np.arange(w)
        flow[:, :, 1] += np.arange(h)[:, np.newaxis]
        mid_frame = cv2.remap(prev, flow, None, cv2.INTER_LINEAR)

        for i in range(1, interpolate_rate+1):
            weight = float(i) / interpolate_rate
            new_frame = cv2.addWeighted(prev, 1-weight, mid_frame, weight, 0)
            iplated.append(Image.fromarray(new_frame))
        prev = frame
 
    return iplated

def viz_flow(im1,flow):
    hsv = np.zeros(im1.shape, dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # plt.imshow(bgr)
    # plt.show()
    return bgr
