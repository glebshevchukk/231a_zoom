import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image, ImageDraw

from interpolate import *
from util import *

IPHONE_K = np.array([[26,0,0],[0,26,0],[0,0,1]])

def run_dolly(config):
    save_path = config.image_dir + config.save_name
    #PREPROCESSING

    li, lid,segd_fg, segd_bg = [],[],[],[]
    images,depth_images = load_images(config.image_dir,config.num_images, config.has_depth,config.extension)
    #load everything in and threshold what's needed
    for image in images:
        l = cv2.imread(image)
        l=cv2.cvtColor(l, cv2.COLOR_BGR2RGB)
        l = cv2.resize(l ,(config.width,config.height)) 
        li.append(l)
    if config.has_depth:
        for i,depth in enumerate(depth_images):
            d = cv2.imread(depth)
            r,d = cv2.threshold(d,config.threshold,1,cv2.THRESH_BINARY)
            d = cv2.resize(d ,(config.width,config.height)) 
            lid.append(d)
            seg = li[i]*d
            seg_b = li[i]*(np.logical_not(d))
            segd_fg.append(seg)
            segd_bg.append(seg_b)
    #get orb features and match them
    final_images = [li[0]]
    if config.has_depth:
        use_for_features = segd_fg
    else:
        use_for_features = li
    for i in range(len(use_for_features)-1):   
        orb = cv2.ORB_create()
        kp1,des1 = orb.detectAndCompute(use_for_features[0],None)
        kp2,des2 = orb.detectAndCompute(use_for_features[i+1],None)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:config.n_best_matches]

        if config.debug:
            matched_img = cv2.drawMatches(use_for_features[0],kp1,use_for_features[i],kp2,matches[:20], None,flags=2)
            plt.imshow(matched_img)
            plt.show()

        #here, we're only applying the change to the background and not the foreground

        if config.algo_type == "affine":
            src_points = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_points = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            X,_ = cv2.estimateAffinePartial2D(dst_points,src_points)

            if config.mapping_type == "separate_background" and config.has_depth:
                foreground = segd_fg[0]
                background = segd_bg[i]
                background = cv2.warpAffine(background, X,(config.width,config.height))
                final_img = cv2.add(foreground,background)
            else:
                final_img = cv2.warpAffine(li[i+1], X,(config.width,config.height))

        elif config.algo_type == "homography":
            src_points = np.int32([kp1[m.queryIdx].pt for m in matches])
            dst_points = np.int32([kp2[m.trainIdx].pt for m in matches])
            H,_ = cv2.findHomography(dst_points,src_points)
            H_inv = np.linalg.inv(H)
            final_img = cv2.warpPerspective(li[i+1],H_inv,(config.width,config.height))
        else:
            print(f"Method {config.algo_type} has not been implemented yet.")
            exit(0)
        final_images.append(final_img)

    #interpolate between all images smoothly
    if config.interpolation_type == 'linear':
        print("Interpolating final video using linear interpolation.")
        iplated = linear_interpolate(final_images)
    else:
        print("Interpolating final video using optical flow interpolation.")
        iplated = flow_interpolate(final_images)

    #perform final saving
    iplated[0].save(save_path,save_all=True, append_images=iplated[1:], optimize=True, duration=500, loop=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--interpolation_type',type=str,default='flow')
    parser.add_argument('--image_dir',type=str,default="img/maggie/")
    parser.add_argument('--save_name',type=str,default="dollied.gif")
    parser.add_argument('--extension',type=str,default=".jpg")
    parser.add_argument('--num_images',type=int,default=4)
    parser.add_argument('--height',type=int,default=1008)
    parser.add_argument('--width',type=int,default=756)
    parser.add_argument('--mapping_type',type=str,default="full_affine")
    parser.add_argument('--algo_type',type=str,default="affine")
    parser.add_argument('--threshold',type=int,default=220)
    parser.add_argument('--n_best_matches',type=int,default=30)
    parser.add_argument('--has_depth',type=bool,default=False)
    parser.add_argument('--debug',type=bool,default=False)
    config = parser.parse_args()
    run_dolly(config)
