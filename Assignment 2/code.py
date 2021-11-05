#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('Image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)

pixel_vals = image.reshape((-1,3))
pixel_vals = np.float32(pixel_vals)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

k = 30
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
 
# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
 
# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))
 
plt.imshow(segmented_image)

#%%

image = cv2.imread('Image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)

pixel_vals = image.reshape((-1,3))

rows = pixel_vals.shape[0]
cols = pixel_vals.shape[1]
data = np.zeros([rows, cols+2])

m = image.shape[0]
n = image.shape[1]
count = 0
for i in range(m):
    for j in range(n):
        data[count][0]=pixel_vals[count][0]
        data[count][1]=pixel_vals[count][1]
        data[count][2]=pixel_vals[count][2]
        data[count][3]=i
        data[count][4]=j
        count+=1

data = data.reshape((-1,5))
data = np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

k = 25
retval, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
 
# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
 

# reshape data into the original image dimensions
segmented_data = np.delete(segmented_data,4,1)
segmented_data = np.delete(segmented_data,3,1)

segmented_image = segmented_data.reshape((image.shape))
 
plt.imshow(segmented_image)