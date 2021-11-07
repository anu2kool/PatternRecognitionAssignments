#Importing necessary libraries for computation and visualisation
import numpy as np
import matplotlib.pyplot as plt
import cv2 #For reading the image data
import kmeans #Importing the kmeans module written in kmeans.py file

#Reading the image from specified location
image = cv2.imread('Image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Converting default BGR to RGB
plt.imshow(image) #Showing the original image

#Flatenning the image data 
pixel_values = image.reshape((-1,3))
pixel_values = np.float32(pixel_values)

data = np.zeros((image.shape[0], image.shape[1], 5))

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        pixel_val = image[i][j]
        data[i][j] = np.array([pixel_val[0]/255, pixel_val[1]/255, pixel_val[2]/255, i/image.shape[0], j/image.shape[1]])

data = data.reshape((-1, 5))

#%%
model_5 = kmeans.KMeans(5)
model_5.fit(data)
fig, ax = plt.subplots(1, 2, figsize=(16, 5))

ax[0].imshow(image)
segmented_5 = model_5.segment_img(data, scale=255).reshape(image.shape)
ax[1].imshow(segmented_5)

