import numpy as np
import matplotlib.pyplot as plt
import cv2 #For reading the image data

"""
KMeans is a class which does image segmentation and can be called from
different programs
"""
class KMeans:
    """
    Initialising the parameters used for the model
    n_clusters -> k value
    max_iter -> maximum number of times we run kmeans
    curr_iter -> current iteration number
    cluster_centers -> the centroids of the data
    """
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.curr_iter = 0
        self.cluster_centers = None
    
    """
    Main logic for training the data using the steps of 
    kmeans 
    """
    def fit(self, X):
        # Randomly assigning the cluster centers 
        indices = np.random.choice(X.shape[0], self.n_clusters)
        self.cluster_centers = np.float32(X[indices]) #typecasting to float datatype
        #Assigning each data point to the nearest centroid
        cluster_assignments = self.assign_clusters(X) 
        #Now it is the iterative part to assigning the points
        #and computes centroids on each iteration and updates
        #it until convergence
        while self.curr_iter < self.max_iter:
            self.curr_iter += 1
            # Updating cluster centers using the mean of cluster points.
            self.update_centers(X, cluster_assignments)

            #Assinging the new clusters to each data point
            new_assignments = self.assign_clusters(X)
            
            # If convergence reached then break
            flag = np.all(new_assignments == cluster_assignments)
            if (flag):
                break
            
            # Updating the cluster assignments.
            cluster_assignments = new_assignments
            
    """
    Updating the cluster centers based on the means of 
    previously allocated data points for each cluster
    """
    def update_centers(self, X, cluster_assignments):
        for i in range(self.n_clusters):
            self.cluster_centers[i] = np.mean(X[cluster_assignments == i], axis=0)   
    
    """
    Assigning each data point the cluster based  on the
    smallest euclidean distance 
    """
    def assign_clusters(self, X):
        cluster_assignments = None
        min_dist = np.array([float('inf')] * X.shape[0])
        for i in range(self.n_clusters):
            # Distance of each point from the ith cluster center.
            dist = np.sqrt(np.sum((X - self.cluster_centers[i])**2, axis=1))
            # Assigning ith cluster to the points where distance is lesser
            # than the previous assignment.
            cluster_assignments = np.where(dist < min_dist, i, cluster_assignments)
            # Updating the minimum distance for points.
            min_dist = np.where(dist < min_dist, dist, min_dist)
        return cluster_assignments
    
    def segment_img(self, img, scale=1):
        #Convert cluster centers to pixel values
        clusters = np.uint8(scale * self.cluster_centers[:, :3])
        # Assign cluster to each pixel.
        cluster_assignments = self.assign_clusters(img)
        # Create segmented image with cluster values as pixel values.
        segmented = clusters[np.uint8(cluster_assignments)]
        return segmented

