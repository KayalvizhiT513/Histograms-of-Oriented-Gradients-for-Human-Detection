import cv2
import numpy as np
import glob

# Path to the images
paths = ["patches/positive patch/*.jpg", "patches/negative patch/*.jpg"]

def gradient(img):
    num_rows, num_cols = img.shape
    
    kx = np.array([[-1, 0, 1]])
    ky = np.array([[-1], [0], [1]])

    gx = np.zeros((num_rows, num_cols))
    gy = np.zeros((num_rows, num_cols))

    gradient_at_pixels = np.zeros((num_rows, num_cols, 2))
    for ridx in range(1, num_rows - 1):
        for cidx in range(1, num_cols - 1):
            gx[ridx, cidx] = np.sum(np.multiply(kx, img[ridx-1:ridx+2, cidx-1:cidx+2]))
            gy[ridx, cidx] = np.sum(np.multiply(ky, img[ridx-1:ridx+2, cidx-1:cidx+2]))

            mag = np.sqrt(gx[ridx, cidx]**2 + gy[ridx, cidx]**2)
            angle = np.arctan2(gy[ridx, cidx], gx[ridx, cidx]) * 180 / np.pi

            gradient_at_pixels[ridx, cidx, 0] = mag
            gradient_at_pixels[ridx, cidx, 1] = angle
    return gradient_at_pixels

gradient_set_of_pos_neg_patches = []

for path in paths:
    images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob(path)]
      
    ret, images  = cv2.imreadmulti(mats=images, 
                                   filename = path, 
                                   flags=0) 
  
    print('Number of images :',len(images))

    gradient_set_of_patches = []

    for image in images:
        gradient_set_of_patches.append(gradient(image))

    gradient_set_of_pos_neg_patches.append(gradient_set_of_patches)

import matplotlib.pyplot as plt

for i in range(len(gradient_set_of_pos_neg_patches)):
    for j in range(len(gradient_set_of_pos_neg_patches[i])):
        gradient_set_of_pos_neg_patches[i][j] = gradient_set_of_pos_neg_patches[i][j].flatten()
        plt.hist(gradient_set_of_pos_neg_patches[i][j], bins=9)
        plt.show()

