# importing necessary libraries
import matplotlib.pyplot as plt  # for plotting and visualising the dataset
import numpy as np  # for mathematical operations
import pandas as pd  # for reading and handling the data
from sklearn.metrics import accuracy_score, confusion_matrix  # for obtaining accuracy score and confusion matrix
from sklearn.model_selection import train_test_split  # for splitting the dataset into training data and test data

COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'purple', 'lime', 'pink', 'yellow', 'orange', 'brown']


class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, viz=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.curr_iter = 0
        self.cluster_centers = None
        self.viz = viz

    def fit(self, X):
        # choosing random points as cluster centers
        indices = np.random.choice(X.shape[0], self.n_clusters)
        self.cluster_centers = np.float32(X[indices])

        # show cluster assignment with random cluster centers
        cluster_assignments = self.assign_clusters(X)
        self.show_clusters(X, cluster_assignments) if self.viz else None

        while self.curr_iter < self.max_iter:
            self.curr_iter += 1

            # update cluster centers using the mean of cluster points
            self.update_centers(X, cluster_assignments)

            # assign new clusters based on the updated cluster centers
            new_assignments = self.assign_clusters(X)

            # if no change in cluster assignment, then exit loop
            flag = np.all(new_assignments == cluster_assignments)
            if flag:
                break

            # update the cluster assignments
            cluster_assignments = new_assignments

            # display the cluster assignments
            self.show_clusters(X, cluster_assignments) if self.viz else None

    def update_centers(self, X, cluster_assignments):
        for i in range(self.n_clusters):
            self.cluster_centers[i] = np.mean(X[cluster_assignments == i], axis=0)

    def show_clusters(self, X, cluster_assignments):
        fig, ax = plt.subplots(1, 1)
        for i in range(self.n_clusters):
            ax.scatter(X[cluster_assignments == i, 0],
                       X[cluster_assignments == i, 1],
                       color=COLORS[i],
                       alpha=0.5)

            ax.scatter(self.cluster_centers[i][0],
                       self.cluster_centers[i][1],
                       color=COLORS[i],
                       s=300,
                       edgecolors='k')
        ax.set_title(f"Clusters after {self.curr_iter} iterations")
        plt.show()

    def assign_clusters(self, X):
        cluster_assignments = None

        min_dist = np.array([float('inf')] * X.shape[0])

        for i in range(self.n_clusters):
            # distance of each point from the ith cluster center
            dist = np.sqrt(np.sum((X - self.cluster_centers[i]) ** 2, axis=1))

            # assign ith cluster to the point if distance is lesser than the previously assigned distance
            cluster_assignments = np.where(dist < min_dist, i, cluster_assignments)

            # update the minimum distance for points
            min_dist = np.where(dist < min_dist, dist, min_dist)

        return cluster_assignments

    def predict(self, x_test, y_test):
        test_pred = list(self.assign_clusters(x_test))

        # Note that the 0/1 class number in the nls_y_test may vary from the 0/1 cluster number from the cluster assignment
        # Code can be run multiple times to verify that in both cases, correct accuracy is being returned
        # Class 0 matches Cluster 0, Class 1 matches Cluster 1
        if test_pred[0] == y_test[0]:
            return test_pred
        # Class 0 matches Cluster 1, Class 1 matches Cluster 0
        else:
            test_pred = [1 if i == 0 else 0 for i in test_pred]
            return test_pred


def solve(data_path):
    # reading the data and storing different classes as class1 and class2
    class1 = pd.read_csv(data_path + "/class1.txt", names=['column1', 'column2'])
    class2 = pd.read_csv(data_path + "/class2.txt", names=['column1', 'column2'])

    # data consists of two features
    # adding the new column which is the label/class for that data point
    # label 0 for class1
    # label 1 for class2
    class1['class'] = 0
    class2['class'] = 1

    # merging the data for both classes
    merged_data = pd.concat([class1, class2])

    # resetting the indices for all the entries
    merged_data = merged_data.reset_index(drop=True)

    # extracting the 'class' column
    y = merged_data['class']

    # dropping the 'class' column
    data = merged_data.drop(['class'], axis=1)

    # splitting the dataset into train and test data in the ratio of 80% to 20%
    X_train, X_test, Y_train, Y_test = (spl_data.to_numpy() for spl_data in train_test_split(data, y, test_size=0.2))

    # initialize KMeans algorithm for two clusters
    kmeans = KMeans(n_clusters=2, viz=True)

    # fit data
    kmeans.fit(X_train)

    # predict classes
    Y_pred = kmeans.predict(X_test, Y_test)

    # show confusion matrix and accuracy over the test data
    print('Confusion Matrix\n', confusion_matrix(Y_test, Y_pred))
    print('Accuracy (in %) = {:.3f}\n'.format(accuracy_score(Y_test, Y_pred) * 100))


# solving for non-linearly separable data
nls_data_path = 'nls_data'
print('Non-linearly Separable Data')
solve(nls_data_path)
