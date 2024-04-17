# Implement Gliding Box Lacunarity Distribution Algorithm
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
                lacunarity_map[i, j] = np.var(unique_labels)/np.mean(unique_labels)
                processed_image[i, j] = np.mean(sub_img)  # Processed image: example using mean

        lacunarity_values.append(np.mean(lacunarity_map))
        processed_images.append(processed_image)

    return lacunarity_values, processed_images
band_number = 2
lacunarity_values, processed_images = gliding_box_lacunarity(selected_band_normalized, box_sizes)
for i, size in enumerate(box_sizes):
    print(f"Gliding Box Lacunarity for Band {band_number} with box size {size}: {lacunarity_values[i]}")

# Display the original band for reference
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(selected_band_normalized, cmap='gray')
plt.title(f'Original Band {band_number}')

# Display the processed images
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
