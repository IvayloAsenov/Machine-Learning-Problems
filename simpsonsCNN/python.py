import numpy as np
np.set_printoptions(precision=2)

import cv2, glob
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils, to_categorical
from keras.optimizers import Adagrad

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import pylab

import itertools

from matplotlib import pyplot as plt

np.random.seed(42)  # for reproducibility

df = pd.read_csv('classes.txt');
df.head();

image_size = 128
labels_name = df.label.unique()

def main():

    pics = []
    labels = []

    for index, row in df.iterrows():
        filepath = './dataset/' + row["filename"]
        temp = cv2.imread(filepath)

        if temp is not None:
            temp = cv2.resize(temp, (image_size, image_size))
            pics.append(temp)
            labels.append(np.where(labels_name == row["label"]))

    input_shape = (image_size, image_size, 3)

    X = np.array(pics)
    y = np.array(labels)

    y = to_categorical(y, len(labels_name)).reshape((len(y), 7))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y,random_state=42);

    # display some samples
    # for idx in np.random.choice(len(X_train), 5):
    #     show_example(idx, X_train, y_train, [])

    # normalize
    X_train_n = X_train.astype('float32')
    X_test_n = X_test.astype('float32')
    X_train_n /= 255
    X_test_n /= 255

    # build our CNN
    model = Sequential()
    model.add(Convolution2D(16, kernel_size=(6,6), padding='same', strides=(3,3), activation='relu', input_shape=input_shape))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    # model summary
    # print(model.summary())

    # train the model
    model.compile(optimizer=Adagrad(), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_n, y_train, validation_data=(X_test_n, y_test), epochs=30, batch_size=12)

    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'b')
    plt.plot(epochs, val_loss_values, 'g')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    pylab.show()

    predictions = model.predict(X_test_n, verbose=0)

    np.argmax(predictions, axis=1)
    np.argmax(y_test, axis=1)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / predictions.shape[0]

    # Let's identify the good predictions and the bad ones
    correct = [idx for idx, correct in enumerate(np.argmax(predictions, axis=1).astype(bool) == np.argmax(y_test, axis=1).astype(bool)) if correct]
    misclassification = [idx for idx, incorrect in enumerate(np.argmax(predictions, axis=1) != np.argmax(y_test, axis=1)) if incorrect]

    for idx in np.random.choice(correct, 5):
        show_example(idx, X_test, y_test, predictions)

    print(len(correct))
    print(len(misclassification))

    for idx in np.random.choice(misclassification, 5):
        show_example(idx, X_test, y_test, predictions)

def show_example(idx, X, y, y_hat):
    """
    idx = index of the images to show
    X = the numpy array of pixel values
    y = the expected label
    y_hat = the predicted label (optional)
    """

    print(f'Sample {idx}')
    if len(y.shape) > 1:
        print(f'Expected is {labels_name[np.argmax(y[idx])]}')
    else:
        print(f'Expected is {labels_name[y[idx]]}')

    if len(y_hat) > 0:
        print(f'Prediction is {labels_name[np.argmax(y_hat[idx])]}')
    plt.imshow(cv2.cvtColor(X[idx], cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == '__main__':
    main()