import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Function to calculate gliding box lacunarity
def gliding_box_lacunarity(data, box_sizes):
    lacunarity_values = []
    processed_images = []

    for size in box_sizes:
        lacunarity_map = np.zeros_like(data)
        processed_image = np.zeros_like(data)

        for i in range(data.shape[0] - size + 1):
            for j in range(data.shape[1] - size + 1):
                sub_img = data[i:i+size, j:j+size]
                unique_labels = np.unique(sub_img)
                if 0 in unique_labels:  # Exclude background label
                    unique_labels = unique_labels[1:]
                lacunarity_map[i, j] = np.var(unique_labels)
                processed_image[i, j] = np.var(sub_img)  # Processed image: example using mean

        lacunarity_values.append(np.mean(lacunarity_map))
        processed_images.append(processed_image)

    return lacunarity_values, processed_images

band_number=2
band_2 = hyperspectral_image[:, :, 2]

height, width = band_2.shape
roi_params = [
    {"x": 0, "y": 0, "roi_width": 50, "roi_height": 50},  # Top-left corner
    {"x": 80, "y": 95, "roi_width": 50, "roi_height": 50},  # Bottom-right corner
    {"x": width - 50, "y": 0, "roi_width": 50, "roi_height": 50},  # Upper-right corner
    {"x": 20, "y": 70, "roi_width": 50, "roi_height": 50}
]
# Set up box sizes for lacunarity calculation
box_sizes = [2, 4, 8, 16, 32]

# Calculate lacunarity for each ROI
for idx, params in enumerate(roi_params):
    x, y, roi_width, roi_height = params["x"], params["y"], params["roi_width"], params["roi_height"]
    roi = band_2[y:y+roi_height, x:x+roi_width]

    # Apply gliding box lacunarity algorithm to the ROI
    lacunarity_values, processed_images = gliding_box_lacunarity(roi, box_sizes)

    # Display results for each ROI
    plt.figure(figsize=(12, 4))
    plt.subplot(1, len(box_sizes) + 1, 1)
    plt.imshow(roi, cmap='gray')
    plt.title(f'ROI {idx+1}')

    for i, size in enumerate(box_sizes):
        plt.subplot(1, len(box_sizes) + 1, i+2)
        plt.imshow(processed_images[i], cmap='gray')
        plt.title(f'Processed Image (Box Size {size})')

    plt.tight_layout()
    plt.show()

    # Display lacunarity values
    plt.plot(box_sizes, lacunarity_values, marker='o')
    plt.xlabel('Box Size')
    plt.ylabel('Gliding Box Lacunarity')
    plt.title(f'Lacunarity for ROI {idx+1}')
    plt.show()

    # Print lacunarity values
    print(f'Lacunarity values for ROI {idx+1}: {lacunarity_values}')

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Function to calculate gliding box lacunarity
def gliding_box_lacunarity(data, box_sizes):
    lacunarity_values = []
    processed_images = []

    for size in box_sizes:
        lacunarity_map = np.zeros_like(data)
        processed_image = np.zeros_like(data)

        for i in range(data.shape[0] - size + 1):
            for j in range(data.shape[1] - size + 1):
                sub_img = data[i:i+size, j:j+size]
                unique_labels = np.unique(sub_img)
                if 0 in unique_labels:  # Exclude background label
                    unique_labels = unique_labels[1:]
                lacunarity_map[i, j] = np.var(unique_labels)
                processed_image[i, j] = np.var(sub_img)  # Processed image: example using mean

        lacunarity_values.append(np.mean(lacunarity_map))
        processed_images.append(processed_image)

    return lacunarity_values, processed_images

band_number=2
band_2 = hyperspectral_image[:, :, 2]
height, width = band_2.shape

roi_params = [
    {"x": 0, "y": 0, "roi_width": 50, "roi_height": 50},  # Top-left corner
    {"x": 80, "y": 95, "roi_width": 50, "roi_height": 50},  # Bottom-right corner
    {"x": width - 50, "y": 0, "roi_width": 50, "roi_height": 50},  # Upper-right corner
    {"x": 20, "y": 70, "roi_width": 50, "roi_height": 50}
]

# Set up box sizes for lacunarity calculation
box_sizes = [2, 4, 8, 16, 32]

# Initialize lists to store lacunarity values for each ROI
all_lacunarity_values = []

# Calculate lacunarity for each ROI
for idx, params in enumerate(roi_params):
    x, y, roi_width, roi_height = params["x"], params["y"], params["roi_width"], params["roi_height"]
    roi = band_2[y:y+roi_height, x:x+roi_width]

    # Apply gliding box lacunarity algorithm to the ROI
    lacunarity_values, processed_images = gliding_box_lacunarity(roi, box_sizes)

    # Store lacunarity values for each ROI
    all_lacunarity_values.append(lacunarity_values)

# Plot all lacunarity values together
plt.figure(figsize=(12, 8))

for idx, lacunarity_values in enumerate(all_lacunarity_values):
    plt.plot(box_sizes, lacunarity_values, marker='o', label=f'ROI {idx+1}')

plt.xlabel('Box Size')
plt.ylabel('Gliding Box Lacunarity')
plt.title('Lacunarity for Different ROIs')
plt.legend()
plt.show()