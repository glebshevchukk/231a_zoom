import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

#assuming for now that images are taken in reverse-z (closer to farther strictly)
image_dir = "img/tonys/"
images, depth_images = [], []
num_images = 3
n_best_matches=5
interpolate_rate = 10
width=1008
height=756

for i in range(num_images):
    images.append(image_dir+str(i)+".jpg")
    depth_images.append(image_dir+str(i)+"_depth.jpg")
threshold = 220



def run_dolly(save_path=image_dir+"dollied.gif"):
    final_images = []
    li, lid,segd = [],[],[]
    #load everything in and threshold what's needed
    for image in images:
        l = cv2.imread(image)
        l=cv2.cvtColor(l, cv2.COLOR_BGR2RGB)
        l = cv2.resize(l ,(height,width)) 
        li.append(l)
    for i,depth in enumerate(depth_images):
        d = cv2.imread(depth)
        r,d = cv2.threshold(d,threshold,1,cv2.THRESH_BINARY)
        d = cv2.resize(d ,(height,width)) 
        lid.append(d)
        seg = li[i]*d
        segd.append(seg)
    
    final_images.append(li[0])

    #get orb features and match them
    #adapting from this cv2 tutorial: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html#flann-based-matcher
    #and https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
    for i,rest_seg in enumerate(segd[1:]):   
        orb = cv2.ORB_create()
        kp1,des1 = orb.detectAndCompute(segd[0],None)
        kp2,des2 = orb.detectAndCompute(segd[i],None)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:n_best_matches]


        src_points = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_points = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        X,_ = cv2.estimateAffinePartial2D(dst_points,src_points)
        final_img = cv2.warpAffine(li[i], X,(height,width))
        final_images.append(final_img)

    #interpolate between all images smoothly
    iplated = []
    prev = final_images[0]
    for frame in final_images[1:]:
        for i in range(1,interpolate_rate):
            weight = float(i) / interpolate_rate
            mid_frame = cv2.addWeighted(prev,weight,frame,1-weight,0) 
            iplated.append(Image.fromarray(mid_frame))
        prev = frame
    iplated[0].save(save_path,
               save_all=True, append_images=iplated[1:], optimize=True, duration=500, loop=1)
    



    #now we want to keep the features in the same spots
    #add a nonlinear transform to each image 


if __name__ == "__main__":
    run_dolly()
