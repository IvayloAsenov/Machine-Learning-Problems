import numpy as np
import cv2, glob

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils, to_categorical
from keras.optimizers import Adagrad

from matplotlib import pyplot as plt
import pylab

from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

np.random.seed(42)

pics = []
labels = []
labels_name = ['Mr Burns', 'Abraham Simpson']

# one epoch  = one forward pass and one backward pass of all the training examples
# batch size = the number of training examples in one forward/backward pass. The higher the
#              batch size, the more memory space you'll need
# number of iterations = number of passes, each pass using [batch size] number of examples
#                        to be clear, one pass = one forward pass + one backward pass

def main():

    for pic in glob.glob('dataset/*.jpg'):
        temp = cv2.imread(pic)
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY);
        temp = cv2.resize(temp, (28,28))
        pics.append(temp)
        labels.append('grandpa' in pic)

    X = np.array(pics)
    y = np.array(labels)

    y = to_categorical(y, 2);

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42);

    # for idx in np.random.choice(len(X_train), 5):
    #     show_example(idx, X_train, y_train, [])

    # Pre Process Data

    # reshape
    X_train_v = X_train.reshape(X_train.shape[0], 28 * 28);
    X_test_v = X_test.reshape(X_test.shape[0], 28 * 28);

    # normalize
    X_train_v = X_train_v.astype('float32')
    X_test_v = X_test_v.astype('float32')
    X_train_v /= 255
    X_test_v /= 255

    # print(X_train_v.shape
    # print(X_test_v.shape)

    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=(28*28)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8,  activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(2, activation='softmax'))

    # print(model.summary())

    model.compile(optimizer=Adagrad(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    # train our model
    history = model.fit(X_train_v, y_train, validation_split=0.05, epochs=100, batch_size=12)

    # learning curve
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'b')
    plt.plot(epochs, val_loss_values, 'g')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    pylab.show()

    # Compute accuracy
    predictions = model.predict(X_test_v, verbose = 0)

    # Accuracy = Number of good predictions / Total number of predictions
    np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / predictions.shape[0]

    # Let's identify the good predictions and the bad ones
    correct = [idx for idx, correct in enumerate(np.argmax(predictions, axis=1).astype(bool) == np.argmax(y_test, axis=1).astype(bool)) if correct]
    misclassification = [idx for idx, incorrect in enumerate(np.argmax(predictions, axis=1) != np.argmax(y_test, axis=1)) if incorrect]

    for idx in np.random.choice(correct, 5):
        show_example(idx, X_test, y_test, predictions)

    # joblib.dump(model, 'filename.pkl');

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
    plt.imshow(X[idx])
    plt.show()

if __name__ == '__main__':
    main()
