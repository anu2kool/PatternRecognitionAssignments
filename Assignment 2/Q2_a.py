#Importing necessary libraries for computation and visualisation
import numpy as np
import matplotlib.pyplot as plt
import cv2 #For reading the image data

#Reading the image from specified location
image = cv2.imread('Image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Converting default BGR to RGB
plt.imshow(image) #Showing the original image

#Flatenning the image data and converting to float data type
pixel_values = image.reshape((-1,3))
pixel_values = np.float32(pixel_values)

#Defining criteria for the Kmeans model with given accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

#Setting the value of k which is the number of clusters
k = 10
#Doing Kmeans and getting centers
retval, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
 
#convert data into 8-bit values
centers = np.uint8(centers)
#getting the flatenned clustered data
segmented_data = centers[labels.flatten()]
 
#reshaping data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))
#Showing the clusters 
plt.imshow(segmented_image)
plt.title("K={}".format(k))