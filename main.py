# -*- coding: utf-8 -*-
"""Paper.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14bnrejB5na0jvMaVrQ5HHMly8DDJYsGa
"""

!pip install np_utils

!pip install spectral

# Commented out IPython magic to ensure Python compatibility.
import keras
import tensorflow as tf
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, BatchNormalization
from keras.layers import Dropout, Input
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import np_utils

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

from operator import truediv

from plotly.offline import init_notebook_mode

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import spectral

init_notebook_mode(connected=True)
# %matplotlib inline

if not (os.path.isfile('Indian_pines_corrected.mat')):
  !wget http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat
if not (os.path.isfile('Indian_pines_gt.mat')):
  !wget http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat

#if not (os.path.isfile('/content/Salinas_corrected.mat')):
#  !wget https://github.com/gokriznastic/HybridSN/raw/master/data/Salinas_corrected.mat
#if not (os.path.isfile('/content/Salinas_gt.mat')):
#  !wget https://github.com/gokriznastic/HybridSN/raw/master/data/Salinas_gt.mat

## GLOBAL VARIABLES
dataset = 'IP'
test_ratio = 0.7
windowSize = 25

def loadData(name):
    data_path = os.path.join(os.getcwd(),'')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']

    return data, labels

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

X, y = loadData(dataset)

X.shape, y.shape

K = X.shape[2]

K = 30 if dataset == 'IP' else 15
X,pca = applyPCA(X,numComponents=K)

X.shape

X, y = createImageCubes(X, y, windowSize=windowSize)

X.shape, y.shape

Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, test_ratio)

Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape

Xtrain = Xtrain.reshape(-1, windowSize, windowSize, K, 1)
Xtrain.shape

ytrain = keras.utils.to_categorical(ytrain)
ytrain.shape

S = windowSize
L = K
output_units = 9 if (dataset == 'PU' or dataset == 'PC') else 16

## input layer
input_layer = Input((S, S, L, 1))

## convolutional layers
conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')(input_layer)
conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(conv_layer1)
conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)
#print(conv_layer3._keras_shape)
conv3d_shape = conv_layer3.shape
conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)
conv_layer4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv_layer3)

flatten_layer = Flatten()(conv_layer4)

## fully connected layers
dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
dense_layer1 = Dropout(0.4)(dense_layer1)
dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
dense_layer2 = Dropout(0.4)(dense_layer2)
output_layer = Dense(units=output_units, activation='softmax')(dense_layer2)

# define the model with input layer and output layer
model = Model(inputs=input_layer, outputs=output_layer)

model.summary()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# compiling the model
#adam = Adam(learning_rate=0.001, decay=1e-06)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# checkpoint
filepath = "best-model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(x=Xtrain, y=ytrain, batch_size=256, epochs=10, callbacks=callbacks_list)

model.save("best-model.hdf5")

# load best weights
model.load_weights("best-model.hdf5")
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

Xtest = Xtest.reshape(-1, windowSize, windowSize, K, 1)
Xtest.shape

ytest = keras.utils.to_categorical(ytest)
ytest.shape

Y_pred_test = model.predict(Xtest)
y_pred_test = np.argmax(Y_pred_test, axis=1)

classification = classification_report(np.argmax(ytest, axis=1), y_pred_test)
print(classification)

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def reports (X_test,y_test,name):
    #start = time.time()
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    #end = time.time()
    #print(end - start)
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
                        ,'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth',
                        'Stubble','Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk',
                        'Vinyard_untrained','Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted metal sheets','Bare Soil','Bitumen',
                        'Self-Blocking Bricks','Shadows']

    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
    oa = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)
    score = model.evaluate(X_test, y_test, batch_size=32)
    Test_Loss =  score[0]*100
    Test_accuracy = score[1]*100

    return classification, confusion, Test_Loss, Test_accuracy, oa*100, each_acc*100, aa*100, kappa*100

classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(Xtest,ytest,dataset)
classification = str(classification)
confusion = str(confusion)
file_name = "classification_report.txt"

with open(file_name, 'w') as x_file:
    x_file.write('{} Test loss (%)'.format(Test_loss))
    x_file.write('\n')
    x_file.write('{} Test accuracy (%)'.format(Test_accuracy))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('{} Kappa accuracy (%)'.format(kappa))
    x_file.write('\n')
    x_file.write('{} Overall accuracy (%)'.format(oa))
    x_file.write('\n')
    x_file.write('{} Average accuracy (%)'.format(aa))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('{}'.format(classification))
    x_file.write('\n')
    x_file.write('{}'.format(confusion))

def Patch(data,height_index,width_index):
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]

    return patch

# load the original image
X, y = loadData(dataset)

height = y.shape[0]
width = y.shape[1]
PATCH_SIZE = windowSize
numComponents = K

X,pca = applyPCA(X, numComponents=numComponents)

X = padWithZeros(X, PATCH_SIZE//2)

# calculate the predicted image
outputs = np.zeros((height,width))
for i in range(height):
    for j in range(width):
        target = int(y[i,j])
        if target == 0 :
            continue
        else :
            image_patch=Patch(X,i,j)
            X_test_image = image_patch.reshape(1,image_patch.shape[0],image_patch.shape[1], image_patch.shape[2], 1).astype('float32')
            prediction = (model.predict(X_test_image))
            prediction = np.argmax(prediction, axis=1)
            outputs[i][j] = prediction+1

ground_truth = spectral.imshow(classes = y,figsize =(7,7))

predict_image = spectral.imshow(classes = outputs.astype(int),figsize =(7,7))

spectral.save_rgb("predictions.jpg", outputs.astype(int), colors=spectral.spy_colors)

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Load Indian Pines dataset from MATLAB file
indian_pines_data = loadmat('/content/Indian_pines_corrected.mat')  # Update the path
indian_pines = indian_pines_data['indian_pines_corrected']  # Adjust based on the structure of your .mat file

# Reshape the data to (num_pixels, num_bands) for easier computation
reshaped_data = np.reshape(indian_pines, (indian_pines.shape[0] * indian_pines.shape[1], indian_pines.shape[2]))

# Calculate the standard deviation for each band
band_std_dev = np.std(reshaped_data, axis=0)

# Find the band with the maximum standard deviation
max_info_band = np.argmax(band_std_dev)

# Plot the standard deviation values for each band
plt.plot(band_std_dev)
plt.xlabel('Band Index')
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation of Bands')
plt.show()

print(f"The band with maximum information is Band {max_info_band + 1}")

predict_image

import numpy as np
from skimage import io
import matplotlib.pyplot as plt

# Load your hyperspectral image
image_path = '/content/predictions.jpg'  # Replace with the path to your image
hyperspectral_image = io.imread(image_path)

# Calculate the standard deviation for each band
std_dev_per_band = np.std(hyperspectral_image, axis=(0, 1))

# Find the band with the maximum standard deviation
max_info_band = np.argmax(std_dev_per_band)

# Plot the standard deviation values for each band
plt.plot(std_dev_per_band)
plt.xlabel('Band Index')
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation for Each Band')
plt.show()

print(f'The band with maximum information is Band {max_info_band + 1}')

print(band_image.shape)

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
