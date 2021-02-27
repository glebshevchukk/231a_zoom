import cv2
import matplotlib.pyplot as plt
import numpy as np

#assuming for now that images are taken in reverse-z (closer to farther strictly)
images = ["img/0.jpg","img/1.jpg","img/2.jpg","img/3.jpg","img/4.jpg"]
depth_images = ["img/0_depth.jpg","img/1_depth.jpg","img/2_depth.jpg","img/3_depth.jpg","img/4_depth.jpg"]
threshold = 200

def run_dolly():
    li, lid,segd = [],[],[]
    #load everything in and threshold what's needed
    for image in images:
        l = cv2.imread(image)
        li.append(l)
    for i,depth in enumerate(depth_images):
        d = cv2.imread(depth)
        r,d = cv2.threshold(d,threshold,1,cv2.THRESH_BINARY)
        d = cv2.resize(d ,(3024,4032)) 
        lid.append(d)
        seg = li[i]*d
        segd.append(seg)

    #get orb features and match them
    #adapting from this cv2 tutorial: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html#flann-based-matcher
    #and https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
    for i,rest_seg in enumerate(segd[1:]):   
        orb = cv2.ORB_create()
        kp1,des1 = orb.detectAndCompute(segd[0],None)
        kp2,des2 = orb.detectAndCompute(rest_seg,None)

        des1,des2 = np.float32(des1),np.float32(des2)

        # FLANN parameters
        flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        matches = flann.knnMatch(des1,des2,2)
        
        thresh = 0.7
        #these will store the matching x,y points between the two cameras
        best_1 = []
        best_2 = []
        for m1,m2 in matches:
            if m1.distance < thresh * m2.distance:
                best_1.append([kp1[m2.queryIdx].pt[0],kp1[m2.queryIdx].pt[1]])
                best_2.append([kp2[m1.trainIdx].pt[0],kp2[m1.trainIdx].pt[1]])
        
        #now, we want to be able to project points from the second image into the first image
        #using a projective transform
        #Ax_2 = x_1

        best_1 = np.array(best_1)
        best_2 = np.array(best_2)

        X,_ = cv2.estimateAffinePartial2D(best_2,best_1)
        
        final_img = cv2.warpAffine(li[i], X,(3024,4032))
        plt.imshow(final_img)
        plt.show()
    



    #now we want to keep the features in the same spots
    #add a nonlinear transform to each image 


if __name__ == "__main__":
    run_dolly()
