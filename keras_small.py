import numpy as np
from scipy.misc import imread
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils

# train image data read and conversion to ndarray of dimensions (no. of images, width, height, channels)

def create_pd_input(path):
    with open(path, 'r') as train_file:
        image_files_names = train_file.readlines()
    for i in range(0, len(image_files_names)):
        image_files_names[i] = image_files_names[i].rstrip('\n')

    pixel_data = []
    for name in image_files_names:
        pixel_data.append(imread('./data/images/' + name + '.png', mode='RGB'))
    pixel_data = np.asarray(pixel_data)
    return pixel_data.astype('float32') / 255.0

# test label data read and conversion to one hot encoding

def process_labels(path):
    
    '''
    Read the input path for label file and return the OHE for labels
    '''
    with open(path, 'r') as label_file:
        labels = label_file.readlines()
    for i in range(0, len(labels)):
        labels[i] = labels[i].rstrip('\n')

    #one hot encoding
    labels_ohe = np_utils.to_categorical(labels, nb_classes=10)
    return labels_ohe

X_pd_train = create_pd_input('./data/X_small_train.txt')
X_pd_test = create_pd_input('./data/X_small_test.txt')

label_train_ohe = process_labels('./data/y_small_train.txt')
label_test_ohe = process_labels('./data/y_small_test.txt')

def create_model():
    cnn_model = Sequential()
    cnn_model.add(Convolution2D(32, 3, 3, activation='relu', dim_ordering='tf', border_mode='same', input_shape=(32, 32, 3), W_constraint=maxnorm(3), bias=True))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3), bias=True))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(512, activation='relu',W_constraint=maxnorm(3)))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(10, activation='softmax'))
    
    sgd = SGD(lr=0.01, momentum=0.9, decay=(0.01/10), nesterov=False)
    cnn_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return cnn_model

model = create_model()
model.fit(X_pd_train, label_train_ohe, validation_data=(X_pd_test, label_test_ohe), nb_epoch=100,          batch_size=10, verbose=2)
acc = model.evaluate(X_pd_test, label_test_ohe, verbose=0)
print 'accuracy %.2f' % (acc[1]*100)
