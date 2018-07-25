from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout

# Build the CNN

classifier = Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(64, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(128, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dropout(0.5))

classifier.add(Dense(output_dim = 6, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

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

classifier.fit_generator(generator = training_set,
                         epochs = 30,
                         validation_data = test_set)


# Testing on new data 

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('data/new-test-images/sideshow_bob_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)
training_set.class_indices
