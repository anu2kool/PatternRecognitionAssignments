# importing necessary libraries
import matplotlib.pyplot as plt  # for plotting and visualising the dataset
import numpy as np  # for mathematical operations
import pandas as pd  # for reading and handling the data
from sklearn.cluster import KMeans  # import k-means cluster library
from sklearn.metrics import accuracy_score, confusion_matrix  # for obtaining accuracy score and confusion matrix
from sklearn.model_selection import train_test_split  # for splitting the dataset into training data and test data


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
    X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size=0.2)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train)
    Y_pred = kmeans.predict(X_test)
    print('Confusion Matrix\n', confusion_matrix(Y_test, Y_pred))
    print('Accuracy (in %) = {:.3f}\n'.format(accuracy_score(Y_test, Y_pred) * 100))


# solving for non-linearly separable data
nls_data_path = 'nls_data'
print('Non-linearly Separable Data')
solve(nls_data_path)
