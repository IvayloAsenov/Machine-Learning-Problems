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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Making the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim=9))
model.add(Dense(6, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size = 10, epochs = 100)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)