import cv2
import numpy as np

# Read the image
image = cv2.imread('frames/seq_001056.jpg')

# Get image dimensions
height, width, _ = image.shape

# Define tile dimensions
tile_height = 80
tile_width = 128

# Calculate the number of tiles in x and y direction
num_tiles_h = height // tile_height
num_tiles_w = width // tile_width

# Create an empty list to store the tiles
tiles = []

# Extract tiles by moving throught the image
for i in range(num_tiles_h):
    for j in range(num_tiles_w):
        # Calculate coordinates
        y1 = i * tile_height
        y2 = (i + 1) * tile_height
        x1 = j * tile_width
        x2 = (j + 1) * tile_width

        # Extract the tile
        tile = image[y1:y2, x1:x2]
        tiles.append(tile)

# Save the tiles
for i, tile in enumerate(tiles):
    cv2.imwrite(f'patches/tile_{i+210}.jpg', tile)  
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
