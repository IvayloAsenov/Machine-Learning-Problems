import numpy as np
np.set_printoptions(precision=2)

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.optimizers import Adagrad

from matplotlib import pyplot as plt
import pylab

from sklearn.metrics import confusion_matrix
import itertools

np.random.seed(123)  # for reproducibility

def main():
    # Load pre-shuffled MNIST data into train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # input image dimensions
    img_rows, img_cols = 28, 28

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    # normalize the features
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert to array of size 10
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    # create the CNN
    model = Sequential()
    model.add(Convolution2D(16, kernel_size=(3,3), activation='relu', input_shape=input_shape))
    model.add(Convolution2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(12, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    print(model.summary())

    # training the network
    model.compile(optimizer=Adagrad(), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=8, batch_size=128)

    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'b')
    plt.plot(epochs, val_loss_values, 'g')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    pylab.show()

    # predictions
    predictions = model.predict(X_test_v, verbose=0)
    
    np.argmax(predictions, axis=1)
    np.argmax(y_test, axis=1)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / predictions.shape[0]

    print(accuracy)

def show_example(idx, X, y, y_hat):
    print(f'Sample {idx}')
    if len(y.shape) > 1:
        print(f'Expected is {np.argmax(y[idx])}')
    else:
        print(f'Expected is {y[idx]}')

    if len(y_hat) > 0:
        print(f'Prediction is {np.argmax(y_hat[idx])}')
    plt.imshow(X[idx].reshape((28,28)))
    plt.show()

if __name__ == '__main__':
    main()
