import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
train_dataset = pd.read_csv('train.csv')
test_dataset  = pd.read_csv('test.csv')

# Fill in missing values
train_dataset['Age'].fillna(train_dataset['Age'].mean(), inplace=True)

X = train_dataset.iloc[:, [2, 4, 5, 6, 7, 9, 11]]
y = train_dataset.iloc[:, 1]

# Encoding categorical data
X = pd.get_dummies(X, columns=['Pclass', 'Sex', 'Embarked'], drop_first = True)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)