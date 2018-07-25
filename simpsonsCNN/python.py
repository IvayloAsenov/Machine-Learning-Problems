from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout

# Build the CNN

classifier = Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dropout(0.5))

classifier.add(Dense(output_dim = 7, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data/training',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = train_datagen.flow_from_directory('data/test',
                                             target_size = (64, 64),
                                             batch_size = 32,
                                             class_mode = 'categorical')

classifier.fit_generator(training_set,
                         epochs = 25,
                         validation_data = test_set)



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
