# importing necessary libraries
import numpy as np  # for mathematical operations
import pandas as pd  # for reading and handling the data
import matplotlib.pyplot as plt  # for plotting and visualising the dataset
from sklearn.metrics import accuracy_score, confusion_matrix  # for obtaining accuracy score and confusion matrix
from sklearn.model_selection import train_test_split  # for splitting the dataset into training data and test data


def solve(data_path):
    # reading the data and storing different classes as class1 and class2
    class1 = pd.read_csv(data_path + "/class1.txt", delimiter=' ', names=['column1', 'column2'])
    class2 = pd.read_csv(data_path + "/class2.txt", delimiter=' ', names=['column1', 'column2'])
    class3 = pd.read_csv(data_path + "/class3.txt", delimiter=' ', names=['column1', 'column2'])

    # clean data
    class1 = class1.reset_index()
    class1.drop('column2', inplace=True, axis=1)
    class1.rename(columns={'index': 'column1', 'column1': 'column2'}, inplace=True)

    class2 = class2.reset_index()
    class2.drop('column2', inplace=True, axis=1)
    class2.rename(columns={'index': 'column1', 'column1': 'column2'}, inplace=True)

    class3 = class3.reset_index()
    class3.drop('column2', inplace=True, axis=1)
    class3.rename(columns={'index': 'column1', 'column1': 'column2'}, inplace=True)

    # data consists of three features
    # adding the new column which is the label/class for that data point
    # label 0 for class1
    # label 1 for class2
    # label 2 for class3
    class1['class'] = 0
    class2['class'] = 1
    class3['class'] = 2

    # merging the data for both classes
    merged_data = pd.concat([class1, class2, class3])

    # resetting the indices for all the entries
    merged_data = merged_data.reset_index(drop=True)

    # extracting the 'class' column
    y = merged_data['class']

    # dropping the 'class' column
    data = merged_data.drop(['class'], axis=1)

    # splitting the dataset into train and test data in the ratio of 80% to 20%
    X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size=0.2)

    # getting the unique class labels and its count
    classes = np.unique(y)
    count = len(classes)

    def get_params(vectors, target):
        """ Compute the parameters of Gaussian distribution.
        """
        mean = vectors.groupby(target).apply(np.mean).to_numpy()
        var = vectors.groupby(target).apply(np.var).to_numpy()
        return mean, var

    def get_likelihood(x, mean, var):
        """ Compute the likelihood of data vector x from the given parameters of Gaussian distribution.
        """
        return np.exp(-(x - mean) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)

    def get_priors(vectors, target):
        """ Compute the prior for each class.
        """
        rows = vectors.shape[0]
        prior = (vectors.groupby(target).apply(lambda x: len(x)) / rows).to_numpy()
        return prior

    def get_posteriors(x, prior, mean, var):
        """ Compute the posterior probability for each class.
        """
        posteriors = []
        for i in range(count):
            priori = np.log(prior[i])
            conditional = np.sum(np.log(get_likelihood(x, mean[i], var[i])))
            posterior = priori + conditional
            posteriors.append(posterior)
        return posteriors

    def get_predictions(vectors, prior, mean, var):
        """ Compute the prediction for each data vector.
        """
        predictions = [classes[np.argmax(get_posteriors(vector, prior, mean, var))] for vector in vectors.to_numpy()]
        return predictions

    mean, var = get_params(X_train, Y_train)
    priors = get_priors(X_train, Y_train)
    Y_predict = get_predictions(X_test, priors, mean, var)

    print('Confusion Matrix\n', confusion_matrix(Y_test, Y_predict))
    print('Accuracy (in %) = {:.3f}'.format(accuracy_score(Y_test, Y_predict) * 100, '\n'))


# solving for real world data
real_world_data_path = 'real_world_data'
print('Real World Data')
solve(real_world_data_path)
