import cv2
import numpy as np
import glob

# Path to the images
path = "patches/positive patch/*.jpg"

# Read all image files
images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob(path)]
  
ret, images  = cv2.imreadmulti(mats=images, 
                               filename = path, 
                               flags=0) 
  
print('Number of images :',len(images))

gamma_corrected_tiles = []

for image in images:
    height, width = image.shape
    image = image.astype(np.float32) / 255.0
    sqrt_gamma_tile = np.sqrt(image)
    gamma_corrected_tile = (sqrt_gamma_tile * 255).astype(np.uint8)
    gamma_corrected_tiles.append(gamma_corrected_tile)

for i, tile in enumerate(gamma_corrected_tiles):
    cv2.imwrite(f'gamma_corrected/positive patch/gamma_corrected_tile_{i}.jpg', tile)
