from __future__ import print_function
import numpy as np
from common import Sketcher
import cv2 as cv
from matplotlib import pyplot as plt

'''
Watershed segmentation
=========
This program demonstrates the watershed segmentation algorithm
in OpenCV: watershed().
Usage
-----
watershed.py [image filename]
Keys
----
  1-7   - switch marker color
  SPACE - update segmentation
  r     - reset
  a     - toggle autoupdate
  ESC   - exit
'''


class Interactive_Segmentation:
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
        conditionalArray = (m == 1)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if conditionalArray[i][j]:
                    segmented_image[i][j][:] = np.array([256, 256, 256])

        segmented_image_mask = np.zeros(conditionalArray.shape)
        segmented_image_mask[conditionalArray == True] = 255
        segmented_image_mask[conditionalArray == False] = 0
        self.segmented_image_mask = segmented_image_mask
        cv.imshow('watershed', segmented_image)

    def run_segmentation(self):
        while cv.getWindowProperty('img', 0) != -1 or cv.getWindowProperty('watershed', 0) != -1:
            ch = cv.waitKey(50)
            if ch == 27:
                break
            if ch >= ord('1') and ch <= ord('7'):
                self.cur_marker = ch - ord('0')
                print('marker: ', self.cur_marker)
            if ch == ord(' ') or (self.sketch.dirty and self.auto_update):
                self.watershed()
                self.sketch.dirty = False
            if ch in [ord('a'), ord('A')]:
                self.auto_update = not self.auto_update
                print('auto_update if', ['off', 'on'][self.auto_update])
            if ch in [ord('r'), ord('R')]:
                self.markers[:] = 0
                self.markers_vis[:] = self.img
                self.sketch.show()
            if ch == ord('\r'):
                cv.destroyAllWindows()
                segmented_image_alpha = cv.cvtColor(
                    self.img, cv.COLOR_RGB2RGBA)

                segmented_image_alpha[:, :, 3] = self.segmented_image_mask.astype(
                    np.uint8)
                cv.imwrite("segmente_foreground.png", segmented_image_alpha)

                self.segmented_image_mask[self.segmented_image_mask == 255] = 1
                self.segmented_image_mask[self.segmented_image_mask == 0] = 255
                self.segmented_image_mask[self.segmented_image_mask == 1] = 0
                segmented_image_alpha[:, :, 3] = self.segmented_image_mask.astype(
                    np.uint8)
                cv.imwrite("segmente_background.png", segmented_image_alpha)

                cv.imshow("Segmented Image", segmented_image_alpha)

                print("Enter!!")
        cv.destroyAllWindows()


if __name__ == '__main__':
    print(__doc__)
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = 'fruits.jpg'
    Interactive_Segmentation(fn).run_segmentation()
