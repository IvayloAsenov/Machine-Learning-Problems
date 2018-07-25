import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
train_dataset = pd.read_csv('train.csv')
test_dataset  = pd.read_csv('test.csv')

# Fill in missing values
train_dataset['Age'].fillna(train_dataset['Age'].mean(), inplace=True)
train_dataset['Fare'].fillna(train_dataset['Fare'].mean(), inplace=True)
test_dataset['Age'].fillna(test_dataset['Age'].mean(), inplace=True)
test_dataset['Fare'].fillna(test_dataset['Fare'].mean(), inplace=True)

X = train_dataset.iloc[:, [2, 4, 5, 6, 7, 9, 11]]
y = train_dataset.iloc[:, 1]

X_test_data = test_dataset.iloc[:, [1, 3, 4, 5, 6, 8, 10]]

# Encoding categorical data
X = pd.get_dummies(X, columns=['Pclass', 'Sex', 'Embarked'], drop_first = True)
X_test_data = pd.get_dummies(X_test_data, columns=['Pclass', 'Sex', 'Embarked'], drop_first = True)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_test_data = sc.transform(X_test_data)

# Making the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim=9))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size = 24, epochs = 120)
y_pred = model.predict(X_test_data)
y_pred = (y_pred > 0.5)

# K-Fold validation ANN 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Dropout

def build_model():
    model = Sequential()
    model.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim=9))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

classifier = KerasClassifier(build_fn=build_model, batch_size = 24, epochs = 120)
accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout

def build_model(optimizer):
    model = Sequential()
    model.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim=9))
    model.add(Dropout(0.2))
    model.add(Dense(6, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

model = KerasClassifier(build_fn=build_model)
parameters = {'batch_size' : [12, 24],
              'epochs'   : [100, 300],
              'optimizer'  : ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = model, param_grid = parameters, scoring = 'accuracy', cv = 5)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_score = grid_search.best_score_

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred = y_pred * 1


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
