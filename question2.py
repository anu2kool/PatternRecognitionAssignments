#Importing necessary libraries 
import numpy as np #For mathematical operations
import pandas as pd #For reading the data and handling data
import matplotlib.pyplot as plt #Plotting and visualising the dataset
from sklearn.metrics import confusion_matrix #Confusion matrix to get the performance metrics and accuracy
from sklearn.model_selection import train_test_split #For splitting the dataset into training and testing

#Reading the linearly separable data and storing the different classes as class1 and class2
class1 = pd.read_csv("ls_data/class1.txt", names=['column1','column2'])
class2 = pd.read_csv("ls_data/class2.txt", names=['column1','column2'])
#data is consisting of two features

#Adding the new column which is the label/class for that data point
#0 label for class1
#1 label for class2
class1['class'] = 0
class2['class'] = 1

#Merging the data for both classes
merged_data = pd.concat([class1, class2], axis=0)
#Resetting the indices for all the entries
merged_data = merged_data.reset_index(drop=True)
#Extracting the class column
y = merged_data['class']

#Dropping the class column 
data = merged_data.drop(['class'],axis = 1)
#Getting the total class labels
classes = np.unique(y)
number_of_classes = len(classes)
number_of_attrs = data.shape[1]

#Splitting the data set into train and test with 20% test size and 80% train size
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
number_of_rows_train = X_train.shape[0]

#Againg merging the class labels to the features
train_data = pd.concat([X_train,y_train ], axis=1)

#Separating the class datas
train_class_0 =  train_data[train_data['class'] == 0]
train_class_1 =  train_data[train_data['class'] == 1]

#Separating the two features for both the classes
x1_class_0 = train_class_0['column1']
x2_class_0 = train_class_0['column2']

x1_class_1 = train_class_1['column1']
x2_class_1 = train_class_1['column2']

#Generating the histograms for attribute 1 for both the classes
x1_class0, bins1_class0, patches1_class0 = plt.hist(x1_class_0,density=True)
x1_class1, bins1_class1, patches1_class1 = plt.hist(x1_class_1,density=True)
plt.show()

#Generating the histograms for attribute 2 for both the classes
x2_class0, bins2_class0, patches2_class0 = plt.hist(x2_class_0,density=True)
x2_class1, bins2_class1, patches2_class1 = plt.hist(x2_class_1,density=True)
plt.show()


#Now predicting the testing data based on information got from the histograms for each data
"""
In this predict function, we are calculating:
Probability of the interval in which the data point is lying(for both the dimensions);
and then finding resultant probability as product of the two probabilities and finding the
maximum of both the classes.
"""
def predictClass(data):
    x1 = data['column1']
    x2 = data['column2']
    prob1_class_0 = 0 #Probability of dimension1 for class 0
    prob1_class_1 = 0 #Probability of dimension1 for class 1
    prob2_class_0 = 0 #Probability of dimension2 for class 0
    prob2_class_1 = 0 #Probability of dimension2 for class 1
    #Finding the respective probabilities
    for i in range(x1_class0.shape[0]-1):
        if x1 >= bins1_class0[i] and x1 <= bins1_class0[i+1]:
            prob1_class_0 =  x1_class0[i]
    for i in range(x1_class1.shape[0]-1):
        if x1 >= bins1_class1[i] and x1 <= bins1_class1[i+1]:
            prob1_class_1 =  x1_class1[i]
    for i in range(x2_class0.shape[0]-1):
        if x2 >= bins2_class0[i] and x2 <= bins2_class0[i+1]:
            prob2_class_0 =  x2_class0[i]
    for i in range(x2_class1.shape[0]-1):
        if x2 >= bins2_class1[i] and x2 < bins2_class1[i+1]:
            prob2_class_1 =  x2_class1[i]
    #Finding resultant probabilities for both the classes
    Probability_class_0 = prob1_class_0*prob2_class_0
    Probability_class_1 = prob2_class_1*prob1_class_1
    if Probability_class_0 > Probability_class_1:
        return 0
    else:
        return 1

y_predict = []
for i in range(X_test.shape[0]):
    class_pred = predictClass(X_test.iloc[i])
    y_predict.append(class_pred)
    
y_predict = np.array(y_predict)
confusionmatrix = confusion_matrix(y_test,y_predict)
print(confusionmatrix)
    