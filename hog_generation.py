import cv2
import numpy as np
import glob

# Path to the images
paths = ["C:/Users/Kayalvizhi/Downloads/mall_dataset/mall_dataset/patches/positive patch/*.jpg", "C:/Users/Kayalvizhi/Downloads/mall_dataset/mall_dataset/patches/negative patch/*.jpg"]

def gradient(img):
    num_rows, num_cols = img.shape
    
    kx = np.array([[-1, 0, 1]])
    ky = np.array([[-1], [0], [1]])

    # 3x3 Sobel kernels
    # kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

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

# Create spatial cells (8 x 8)
def spatial_cells(gradient_set_of_patches):
    spatial_cells = []
    for patches in gradient_set_of_patches:
        spatial_cells_patches = []
        for patch in patches:
            num_rows, num_cols, _ = patch.shape
            spatial_cell = np.zeros((num_rows//8, num_cols//8, 9))
            for ridx in range(0, num_rows, 8):
                for cidx in range(0, num_cols, 8):
                    for r in range(ridx, ridx + 8):
                        for c in range(cidx, cidx + 8):
                            mag, angle = patch[r, c]
                            angle = angle % 180
                            bin_idx = int(angle // 20)
                            bin_fraction = (angle % 20) / 20
                            # To reduce aliasing, votes are interpolated bilinearly 
                            # between the neighbouring bin centres in both orientation and position.
                            spatial_cell[ridx//8, cidx//8, bin_idx] += mag * (1 - bin_fraction)
                            spatial_cell[ridx//8, cidx//8, (bin_idx + 1) % 9] += mag * bin_fraction
            spatial_cells_patches.append(spatial_cell)
        spatial_cells.append(spatial_cells_patches)
    return spatial_cells

spatial_cells_pos_neg_patches = spatial_cells(gradient_set_of_pos_neg_patches)

# normalize and descriptor blocks
def normalize_descriptor_blocks(spatial_cells_pos_neg_patches):
    # Grouping cells into blocks (2 x 2)
    descriptor_blocks = []
    for spatial_cells_patches in spatial_cells_pos_neg_patches:
        descriptor_blocks_patches = []
        for spatial_cells in spatial_cells_patches:
            num_rows, num_cols, _ = spatial_cells.shape
            # blocks are formed from over-lapping cells thus only 1 row and col reduced
            descriptor_block = np.zeros((num_rows-1, num_cols-1, 36))
            for ridx in range(0, num_rows - 1):
                for cidx in range(0, num_cols - 1):
                    block = spatial_cells[ridx:ridx+2, cidx:cidx+2, :].flatten()
                    eps = 1e-10
                    # L2 norm                    
                    norm_block = block/np.sqrt(np.sum(block**2) + eps)
                    descriptor_block[ridx, cidx] = norm_block
            descriptor_block = descriptor_block.flatten()
            descriptor_blocks_patches.append(descriptor_block)
        descriptor_blocks.append(descriptor_blocks_patches)
    return descriptor_blocks

descriptor_blocks_pos_neg_patches = normalize_descriptor_blocks(spatial_cells_pos_neg_patches)

# Label the data descriptor_blocks_pos_neg_patches[0] as 1 and descriptor_blocks_pos_neg_patches[1] as 0
labels = []
data = []

for i, descriptor_blocks in enumerate(descriptor_blocks_pos_neg_patches):
    label = 1 if i == 0 else 0
    for descriptor_block in descriptor_blocks:
        data.append(descriptor_block)
        labels.append(label)

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Data shape: (30+28, (16-1)*(10-1)*(9bins*4))
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# save the data and label as csv
np.savetxt("data.csv", data, delimiter=",")
np.savetxt("labels.csv", labels, delimiter=",")
print("Data and labels saved successfully!")