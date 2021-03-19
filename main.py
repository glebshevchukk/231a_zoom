'''
Main runner for performing computational dolly zoom.
Resources used:
1) https://github.com/pyimreg/python-register/tree/master/imreg
2) https://www.geeksforgeeks.org/image-registration-using-opencv-python/
3) https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image, ImageDraw

from interpolate import *
from util import *
from spline import *
import scipy.ndimage as nd

# from imreg import model, register
# from imreg.samplers import sampler
from manual_segmentation import *

IPHONE_K = np.array([[26, 0, 0], [0, 26, 0], [0, 0, 1]])


def run_dolly(config):
    save_path = config.image_dir + config.algo_type + "_" + config.save_name
    save_orig_path = config.image_dir + config.save_orig_name
    # PREPROCESSING

    li, lid, segd_fg, segd_bg = [], [], [], []
    images, depth_images, pre_segmented_images = load_images(
        config.image_dir, config.num_images, config.has_depth, config.pre_segmented, config.extension)
    # load everything in and threshold what's needed
    for image in images:
        l = cv2.imread(image)
        l = cv2.cvtColor(l, cv2.COLOR_BGR2RGB)
        l = cv2.resize(l, (config.width, config.height))
        li.append(l)
    if config.has_depth:
        for i, depth in enumerate(depth_images):
            d = cv2.imread(depth)
            r, d = cv2.threshold(d, config.threshold, 1, cv2.THRESH_BINARY)
            d = cv2.resize(d, (config.width, config.height))
            lid.append(d)
            seg = li[i]*d
            seg_b = li[i]*(np.logical_not(d))
            segd_fg.append(seg)
            segd_bg.append(seg_b)
    if config.manual_segmentation:
        for i, image in enumerate(images):
            foreGround, backGround, depthMap = img_seg(
                image, i, config.image_dir).run_segmentation()
            lid.append(depthMap)
            segd_fg.append(foreGround)
            segd_bg.append(backGround)
    if config.pre_segmented:
        for i, image in enumerate(pre_segmented_images):
            d = cv2.resize(cv.imread(image), (config.width, config.height))
            lid.append(d)
            seg = li[i]*d
            seg_b = li[i]*(np.logical_not(d))
            segd_fg.append(seg)
            segd_bg.append(seg_b)

            # get orb features and match them
    final_images = [li[0]]
    if config.has_depth or config.manual_segmentation or config.pre_segmented:
        use_for_features = segd_fg
    else:
        use_for_features = li
    for i in range(len(use_for_features)-1):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(use_for_features[0], None)
        kp2, des2 = orb.detectAndCompute(use_for_features[i+1], None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[
            :config.n_best_matches]

        if config.debug:
            matched_img = cv2.drawMatches(
                use_for_features[0], kp1, use_for_features[i], kp2, matches[:20], None, flags=2)
            plt.imshow(matched_img)
            plt.show()

        # here, we're only applying the change to the background and not the foreground

        if config.algo_type == "affine":
            src_points = np.float32(
                [kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_points = np.float32(
                [kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            X, _ = cv2.estimateAffinePartial2D(dst_points, src_points)

            if config.mapping_type == "separate_background" and (config.has_depth or config.manual_segmentation or config.pre_segmented):
                foreground = segd_fg[0]
                background = segd_bg[i]
                background = cv2.warpAffine(
                    background, X, (config.width, config.height))
                final_img = cv2.add(foreground, background)
            else:
                final_img = cv2.warpAffine(
                    li[i+1], X, (config.width, config.height))

        elif config.algo_type == "homography":
            src_points = np.int32([kp1[m.queryIdx].pt for m in matches])
            dst_points = np.int32([kp2[m.trainIdx].pt for m in matches])
            H, _ = cv2.findHomography(dst_points, src_points, cv2.RANSAC)
            final_img = cv2.warpPerspective(
                li[i+1], H, (config.width, config.height))

        elif config.algo_type == "spline":
            src_points = np.float32([kp1[m.queryIdx].pt for m in matches])
            dst_points = np.float32([kp2[m.trainIdx].pt for m in matches])
            i1 = RegisterData(li[0], features=src_points)
            i2 = RegisterData(li[i+1], features=dst_points)

            feature = register.FeatureRegister(
                model=model.ThinPlateSpline,
                sampler=sampler.Spline,
            )

            # Perform the registration.
            p, warp, final_img, error = feature.register(i1, i2)
            if config.debug:
                plt.imshow(final_img)
                plt.show()
        else:
            print(f"Method {config.algo_type} has not been implemented yet.")
            exit(0)
        final_images.append(final_img)

    # interpolate between all images smoothly
    if config.interpolation_type == 'linear':
        print("Interpolating final video using linear interpolation.")
        iplated = linear_interpolate(final_images)
    else:
        print("Interpolating final video using optical flow interpolation.")
        iplated = flow_interpolate(final_images)

    # perform final saving
    orig = [Image.fromarray(l) for l in li]
    orig[0].save(save_orig_path, save_all=True,
                 append_images=orig[1:], optimize=True, duration=100, loop=0)
    iplated[0].save(save_path, save_all=True, append_images=iplated[1:],
                    optimize=True, duration=100, loop=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--interpolation_type', type=str, default='flow')
    parser.add_argument('--image_dir', type=str, default="img/maggie/")
    parser.add_argument('--save_name', type=str, default="dollied.gif")
    parser.add_argument('--extension', type=str, default=".jpg")
    parser.add_argument('--num_images', type=int, default=4)
    parser.add_argument('--height', type=int, default=1008)
    parser.add_argument('--width', type=int, default=756)
    parser.add_argument('--mapping_type', type=str, default="full_affine")
    parser.add_argument('--algo_type', type=str, default="affine")
    parser.add_argument('--threshold', type=int, default=180)
    parser.add_argument('--n_best_matches', type=int, default=30)
    parser.add_argument('--has_depth', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--manual_segmentation', type=bool, default=False)
    parser.add_argument('--save_orig_name', type=str, default="original.gif")
    parser.add_argument('--pre_segmented', type=bool, default=False)
    config = parser.parse_args()
    run_dolly(config)
