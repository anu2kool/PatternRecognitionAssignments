#Importing necessary libraries for computation and visualisation
import numpy as np
import matplotlib.pyplot as plt
import cv2 #For reading the image data

#Reading the image from specified location
image = cv2.imread('Image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Converting default BGR to RGB
plt.imshow(image) #Showing the original image

#Flatenning the image data 
pixel_values = image.reshape((-1,3))

#Getting the dimensions of the image
rows = pixel_values.shape[0]
cols = pixel_values.shape[1]
#Initialising the data with 2 extra columns for i(row) and j(col) for each pixel adding two new features
data = np.zeros([rows, cols+2])

#Getting the dimensions for data with i and j (pixel location)
m = image.shape[0]
n = image.shape[1]
count = 0
#Flattening the data
for i in range(m):
    for j in range(n):
        data[count][0]=pixel_values[count][0]
        data[count][1]=pixel_values[count][1]
        data[count][2]=pixel_values[count][2]
        data[count][3]=i
        data[count][4]=j
        count+=1

#Reshaping the data and conversion to float datatype
data = data.reshape((-1,5))
data = np.float32(data)

#Defining criteria for the Kmeans model with given accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

#Setting the value of k which is the number of clusters
k = 30
#Doing Kmeans and getting centers
retval, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
 
# convert data into 8-bit values
centers = np.uint8(centers)
#getting the flatenned clustered data
segmented_data = centers[labels.flatten()]
 

#Dropping the last 2 columns of pixel location as we have to plot the image using this data
segmented_data = np.delete(segmented_data,4,1)
segmented_data = np.delete(segmented_data,3,1)

#reshaping data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))
#Showing the clusters
 
plt.imshow(segmented_image)
plt.title("K={}".format(k))