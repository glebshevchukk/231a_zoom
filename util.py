import numpy as np
# assuming for now that images are taken in reverse-z (closer to farther strictly)


def load_images(image_dir, num_images, has_depth, pre_segmented, extension=".jpg"):
    images, depth_images, pre_segmented_imgaes = [], [], []
    for i in range(num_images):
        images.append(image_dir+str(i)+extension)
        if has_depth:
            depth_images.append(image_dir+str(i)+"_depth"+extension)
        if pre_segmented:
            pre_segmented_imgaes.append(
                image_dir+str(i)+"_segmented_depth"+extension)
    return images, depth_images, pre_segmented_imgaes

# adapted from https://medium.com/analytics-vidhya/using-homography-for-pose-estimation-in-opencv-a7215f260fdd


def decompose_homography(H, K):
    K_inv = np.linalg.inv(K)
    L = 1 / np.linalg.norm(np.dot(K_inv, H[0]))
    r1 = L * np.dot(K_inv, H[0])
    r2 = L * np.dot(K_inv, H[1])
    r3 = np.cross(r1, r2)
    T = L * (K_inv @ H[2].reshape(3, 1))
    R = np.array([[r1], [r2], [r3]])
    R = np.reshape(R, (3, 3))

    return R, T
