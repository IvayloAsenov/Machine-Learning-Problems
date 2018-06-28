import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.optimizers import Adagrad

from matplotlib import pyplot as plt
import pylab

np.random.seed(123)  # for reproducibility

def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # for idx in np.random.choice(len(X_train), 5):
    #     show_example(idx, X_train, y_train, [])

    # reshape our images
    X_train_v = X_train.reshape(X_train.shape[0], 28 * 28)
    X_test_v = X_test.reshape(X_test.shape[0], 28 * 28)

    # normalize the features
    X_train_v = X_train_v.astype('float32')
    X_test_v = X_test_v.astype('float32')
    X_train_v /= 255
    X_test_v /= 255

    # convert target to array
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    # create network architecture
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=(28*28)))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(10, activation='softmax'))

    # print(model.summary)

    # training the network
    model.compile(optimizer=Adagrad(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_v, y_train, validation_split=0.05, epochs=50, batch_size=64)

    # stats
    history_dict = history.history
    history_dict.keys()

    # curves
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

    # our predictions
    np.argmax(predictions, axis=1)
    # expected result
    np.argmax(y_test, axis=1)
    # Accuracy
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
    plt.imshow(X[idx])
    plt.show()

if __name__ == '__main__':
    main()
