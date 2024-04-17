import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

band_number=2
band_image = hyperspectral_image[:, :, 2]
selected_band_normalized = StandardScaler().fit_transform(band_image.reshape(-1, 1)).reshape(band_image.shape)
box_sizes = [2, 4, 8, 16, 32]
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
                lacunarity_map[i, j] = np.var(unique_labels)  # Using variance for lacunarity
                processed_image[i, j] = np.mean(sub_img)  # Processed image: example using mean

        lacunarity_values.append(np.mean(lacunarity_map))
        processed_images.append(processed_image)

    return lacunarity_values, processed_images

# Apply the algorithm to the selected band
lacunarity_values, processed_images = gliding_box_lacunarity(selected_band_normalized, box_sizes)

# Display lacunarity values for each box size
for i, size in enumerate(box_sizes):
    print(f"Gliding Box Lacunarity for Band {band_number} with box size {size}: {lacunarity_values[i]}")

# Display the original band for reference
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(selected_band_normalized, cmap='gray')
plt.title(f'Original Band {band_number}')

# Display the processed images
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
plt.title(f'Gliding Box Lacunarity for Band {band_number}')
plt.show()

print(band_image)

pip install numpy opencv-python

band_number=2
band_2 = hyperspectral_image[:, :, 2]
x, y, roi_width, roi_height = 0, 0, 50, 50
# Define ROI parameters (top-left corner coordinates and dimensions)
x, y, roi_width, roi_height = 0, 0, 50, 50  # Adjust these values based on your requirements

# Extract the ROI using NumPy array slicing
roi = band_2[y:y+roi_height, x:x+roi_width]

# Display the original Band 29 image and the extracted ROI
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(band_2, cmap='gray')
plt.title('Original Band 2')

plt.subplot(1, 2, 2)
plt.imshow(roi, cmap='gray')
plt.title('ROI (Upper-Left Corner)')

plt.show()

band_number=2
band_2 = hyperspectral_image[:, :, 2]
# Define ROI parameters (top-left corner coordinates and dimensions)
x, y, roi_width, roi_height = 80, 95, 50, 50  # Adjust these values based on your requirements

# Extract the ROI using NumPy array slicing
roi = band_2[y:y+roi_height, x:x+roi_width]

# Display the original Band 29 image and the extracted ROI
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(band_2, cmap='gray')
plt.title('Original Band 2')

plt.subplot(1, 2, 2)
plt.imshow(roi, cmap='gray')
plt.title('ROI (Bottom-Right Corner)')

plt.show()

band_number=2
band_2 = hyperspectral_image[:, :, 2]
# Define ROI parameters (top-left corner coordinates and dimensions)
roi_width, roi_height = 50, 50  # Adjust these values based on your requirements
x = width - roi_width
y = 0

# Extract the ROI using NumPy array slicing
roi = band_2[y:y+roi_height, x:x+roi_width]

# Display the original Band 29 image and the extracted ROI
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(band_2, cmap='gray')
plt.title('Original Band 2')

plt.subplot(1, 2, 2)
plt.imshow(roi, cmap='gray')
plt.title('ROI (Upper-Right Corner)')

plt.show()

band_number=2
band_2 = hyperspectral_image[:, :, 2]
# Define ROI parameters (top-left corner coordinates and dimensions)
x, y, roi_width, roi_height = 20, 70, 50, 50
# Extract the ROI using NumPy array slicing
roi = band_2[y:y+roi_height, x:x+roi_width]

# Display the original Band 29 image and the extracted ROI
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(band_2, cmap='gray')
plt.title('Original Band 2')

plt.subplot(1, 2, 2)
plt.imshow(roi, cmap='gray')
plt.title('ROI (Randomly Selected)')

plt.show()
