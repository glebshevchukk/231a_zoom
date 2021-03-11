from __future__ import print_function
import numpy as np
from common import Sketcher
import cv2 as cv
from matplotlib import pyplot as plt

'''
Interactive Watershed Segmentation
===

Keys
----
  1 - 2   - switch marker color
    1 = subject
    2 = background
  SPACE - update segmentation
  r     - reset
  ESC   - exit
'''


class img_seg:
    def __init__(self, fn):
        self.img = cv.imread(fn)
        resize = 0.2
        self.img = cv.resize(self.img, None, fx=resize,
                             fy=resize, interpolation=cv.INTER_AREA)
        if self.img is None:
            raise Exception('Failed to load image file: %s' % fn)

        h, w = self.img.shape[:2]
        self.markers = np.zeros((h, w), np.int32)
        self.markers_vis = self.img.copy()
        self.cur_marker = 1
        self.colors = np.int32(list(np.ndindex(2, 2, 2))) * 255

        self.auto_update = True
        self.sketch = Sketcher(
            'img', [self.markers_vis, self.markers], self.get_colors)

    def get_colors(self):
        return list(map(int, self.colors[self.cur_marker])), self.cur_marker

    def watershed(self):
        m = self.markers.copy()
        cv.watershed(self.img, m)
        segmented_image = np.zeros(self.img.shape)

        # 1 displays the foreground
        # 2 displays the background
        # -1 displays the edges
        self.conditionalArray = (m == 1)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if self.conditionalArray[i][j]:
                    segmented_image[i][j][:] = np.array([256, 256, 256])

        segmented_image_mask = np.zeros(self.conditionalArray.shape)
        segmented_image_mask[self.conditionalArray == True] = 255
        segmented_image_mask[self.conditionalArray == False] = 0
        self.segmented_image_mask = segmented_image_mask
        cv.imshow('watershed', segmented_image)

    def run_segmentation(self):
        while cv.getWindowProperty('img', 0) != -1 or cv.getWindowProperty('watershed', 0) != -1:
            ch = cv.waitKey(50)
            if ch == 27:
                break
            if ch >= ord('1') and ch <= ord('2'):
                self.cur_marker = ch - ord('0')
                print('marker: ', self.cur_marker)
            if ch == ord(' ') or (self.sketch.dirty and self.auto_update):
                self.watershed()
                self.sketch.dirty = False
            if ch in [ord('r'), ord('R')]:
                self.markers[:] = 0
                self.markers_vis[:] = self.img
                self.sketch.show()
            if ch == ord('\r'):
                cv.destroyAllWindows()
                segmented_image_alpha_fg = cv.cvtColor(
                    self.img, cv.COLOR_RGB2RGBA)

                segmented_image_alpha_fg[:, :, 3] = self.segmented_image_mask.astype(
                    np.uint8)

                self.segmented_image_mask[self.segmented_image_mask == 255] = 1
                self.segmented_image_mask[self.segmented_image_mask == 0] = 255
                self.segmented_image_mask[self.segmented_image_mask == 1] = 0

                segmented_image_alpha_bg = cv.cvtColor(
                    self.img, cv.COLOR_RGB2RGBA)
                segmented_image_alpha_bg[:, :, 3] = self.segmented_image_mask.astype(
                    np.uint8)

                return segmented_image_alpha_fg, segmented_image_alpha_bg, self.conditionalArray
        cv.destroyAllWindows()
        return 0, 0, 0
