import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image, ImageDraw

#assuming for now that images are taken in reverse-z (closer to farther strictly)

def load_images(image_dir,num_images,has_depth,extension=".jpg"):
    images, depth_images = [], []
    for i in range(num_images):
        images.append(image_dir+str(i)+extension)
        if has_depth:
            depth_images.append(image_dir+str(i)+"_depth"+extension)
    return images,depth_images

def linear_interpolate(images,interpolate_rate=5):
    '''
    Create a video interpolating between the frames by taking a linear
    weight of each pair of neighbor frames
    '''
    assert len(images) > 1
    iplated = []
    prev = images[0]
    for frame in images[1:]:
        for i in range(1,interpolate_rate+1):
            weight = float(i) / interpolate_rate
            mid_frame = cv2.addWeighted(prev,1-weight,frame,weight,0) 
            iplated.append(Image.fromarray(mid_frame))
        prev = frame
    return iplated

def flow_interpolate(images,interpolate_rate=5):
    # Calculates dense optical flow by Farneback method 
    prev = images[0]
    iplated = []
    for frame in images[1:]:
        
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY) 
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
        flow = cv2.calcOpticalFlowFarneback(prev_gray,gray, None, 0.5, 3, 25, 3, 7, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN) 
        h, w = flow.shape[:2]

        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        mid_frame = cv2.remap(prev,flow,None,cv2.INTER_LINEAR)

        for i in range(1,interpolate_rate+1):
           weight = float(i) / interpolate_rate
           new_frame = cv2.addWeighted(prev,1-weight,mid_frame,weight,0) 
           iplated.append(Image.fromarray(new_frame))
        prev = frame
    print(len(iplated))
    return iplated



def run_dolly(config):
    save_path = config.image_dir + config.save_name
    #PREPROCESSING

    li, lid,segd = [],[],[]
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
            r,d = cv2.threshold(d,config.threshold,1,cv2.THRESH_BINARY_INV)
            d = cv2.resize(d ,(config.width,config.height)) 
            lid.append(d)
            seg = li[i]*d
            segd.append(seg)
    #get orb features and match them
    #adapting from this cv2 tutorial: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html#flann-based-matcher
    #and https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html

    final_images = [li[0]]
    if config.has_depth:
        use_for_features = segd
    else:
        use_for_features = li
    for i in range(len(use_for_features)-1):   
        orb = cv2.ORB_create()
        kp1,des1 = orb.detectAndCompute(use_for_features[0],None)
        kp2,des2 = orb.detectAndCompute(use_for_features[i+1],None)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:config.n_best_matches]


        src_points = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_points = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        X,_ = cv2.estimateAffinePartial2D(dst_points,src_points)
        final_img = cv2.warpAffine(li[i+1], X,(config.width,config.height))
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
    parser.add_argument('--threshold',type=int,default=220)
    parser.add_argument('--n_best_matches',type=int,default=30)
    parser.add_argument('--has_depth',type=bool,default=False)
    config = parser.parse_args()
    run_dolly(config)
