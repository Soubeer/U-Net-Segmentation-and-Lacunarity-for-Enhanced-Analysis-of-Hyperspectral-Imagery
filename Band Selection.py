!pip install scikit-image --upgrade

import numpy as np
from skimage.feature import greycomatrix
from skimage import io, color
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy.ndimage import variance


# Load your hyperspectral image
image_path = '/content/predictions.jpg'  # Replace with the path to your image
hyperspectral_image = io.imread(image_path)

# Get the dimensions of the image
num_rows, num_cols, num_bands = hyperspectral_image.shape

# Calculate the lacunarity for each band
lacunarity_per_band = np.zeros(num_bands)

patch_size = 5  # You can adjust this value based on your image characteristics


band_image = hyperspectral_image[:, :, 2]
lacunarity_value = variance(band_image)

# Plot the lacunarity values for each band
plt.imshow(band_image, cmap='gray')
plt.xlabel('Band Index')
plt.ylabel('Lacunarity')
plt.title('Band 2')
plt.show()

# Find the band with the maximum lacunarity
max_lacunarity_band = np.argmax(lacunarity_per_band)

print(f'The band with maximum lacunarity is Band {lacunarity_value}')

from sklearn.preprocessing import StandardScaler

band_image = hyperspectral_image[:, :, 2]
selected_band_normalized = StandardScaler().fit_transform(band_image.reshape(-1, 1)).reshape(band_image.shape)

box_sizes = [2, 4, 8, 16, 32]