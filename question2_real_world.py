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

    # again merging the class labels to the features
    train_data = pd.concat([X_train, Y_train], axis=1)

    # separating the class data
    train_class_0 = train_data[train_data['class'] == 0]
    train_class_1 = train_data[train_data['class'] == 1]

    # separating the two features for both the classes
    x1_class_0 = train_class_0['column1']
    x2_class_0 = train_class_0['column2']

    x1_class_1 = train_class_1['column1']
    x2_class_1 = train_class_1['column2']

    # generating the histograms for attribute 1 for both the classes
    x1_class0, bins1_class0, patches1_class0 = plt.hist(x1_class_0, density=True,
                                                        label='Training Data Class0 Attribute1')
    x1_class1, bins1_class1, patches1_class1 = plt.hist(x1_class_1, density=True,
                                                        label='Training Data Class1 Attribute1')
    plt.legend()
    plt.show()

    # generating the histograms for attribute 2 for both the classes
    x2_class0, bins2_class0, patches2_class0 = plt.hist(x2_class_0, density=True,
                                                        label='Training Data Class0 Attribute2')
    x2_class1, bins2_class1, patches2_class1 = plt.hist(x2_class_1, density=True,
                                                        label='Training Data Class1 Attribute2')
    plt.legend()
    plt.show()

    # now predicting the test data based on information retrieved from the histograms for each data
    """
    In this predict function, we are calculating:
    Probability of the interval in which the data point is lying (for both the dimensions);
    and then finding resultant probability as product of the two probabilities and finding the
    maximum of both the classes.
    """

    def predictClass(data):
        x1 = data['column1']
        x2 = data['column2']
        prob1_class_0 = 0  # Probability of dimension1 for class 0
        prob1_class_1 = 0  # Probability of dimension1 for class 1
        prob2_class_0 = 0  # Probability of dimension2 for class 0
        prob2_class_1 = 0  # Probability of dimension2 for class 1

        # finding the respective probabilities
        for i in range(x1_class0.shape[0] - 1):
            if bins1_class0[i] <= x1 <= bins1_class0[i + 1]:
                prob1_class_0 = x1_class0[i]
        for i in range(x1_class1.shape[0] - 1):
            if bins1_class1[i] <= x1 <= bins1_class1[i + 1]:
                prob1_class_1 = x1_class1[i]
        for i in range(x2_class0.shape[0] - 1):
            if bins2_class0[i] <= x2 <= bins2_class0[i + 1]:
                prob2_class_0 = x2_class0[i]
        for i in range(x2_class1.shape[0] - 1):
            if bins2_class1[i] <= x2 <= bins2_class1[i + 1]:
                prob2_class_1 = x2_class1[i]

        # finding resultant probabilities for both the classes
        Probability_class_0 = prob1_class_0 * prob2_class_0
        Probability_class_1 = prob2_class_1 * prob1_class_1
        if Probability_class_0 > Probability_class_1:
            return 0
        else:
            return 1

    y_predict = []
    for i in range(X_test.shape[0]):
        class_pred = predictClass(X_test.iloc[i])
        y_predict.append(class_pred)

    # show confusion matrix and accuracy of classifier over the test data
    y_predict = np.array(y_predict)
    print('Confusion Matrix\n', confusion_matrix(Y_test, y_predict))
    print('Accuracy (in %) = {:.3f}'.format(accuracy_score(Y_test, y_predict) * 100, '\n'))


# solving for two data sets
real_world_data_path = 'real_world_data'
print('Real World Data')
solve(real_world_data_path)
