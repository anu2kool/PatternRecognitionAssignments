#Importing necessary libraries for computation and visualisation
import numpy as np
import matplotlib.pyplot as plt
import cv2 #For reading the image data
import kmeans #Importing the kmeans module written in kmeans.py file



#Reading the image from specified location
image = cv2.imread('Image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Converting default BGR to RGB

#Flatenning the image data and converting to float data type
pixel_values = image.reshape((-1,3))
pixel_values = np.float32(pixel_values)

#%%
#For K=2
model_2 = kmeans.KMeans(2)
model_2.fit(pixel_values)

fig, ax = plt.subplots(1, 2, figsize=(16, 5))

ax[0].imshow(image)
segmented_2 = model_2.segment_img(pixel_values).reshape(image.shape)
ax[1].imshow(segmented_2)

#%%
#For K=3
model_3 = kmeans.KMeans(3)
model_3.fit(pixel_values)

fig, ax = plt.subplots(1, 2, figsize=(16, 5))

ax[0].imshow(image)
segmented_3 = model_3.segment_img(pixel_values).reshape(image.shape)
ax[1].imshow(segmented_3)

#%%
#For K=4
model_4 = kmeans.KMeans(4)
model_4.fit(pixel_values)

fig, ax = plt.subplots(1, 2, figsize=(16, 5))

ax[0].imshow(image)
segmented_4 = model_4.segment_img(pixel_values).reshape(image.shape)
ax[1].imshow(segmented_4)

#%%
#For K=5
model_5 = kmeans.KMeans(5)
model_5.fit(pixel_values)

fig, ax = plt.subplots(1, 2, figsize=(16, 5))

ax[0].imshow(image)
segmented_5 = model_5.segment_img(pixel_values).reshape(image.shape)
ax[1].imshow(segmented_5)
