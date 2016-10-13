import numpy as np
from scipy.misc import imread
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Convolution2D
from keras.layers import SeparableConvolution2D
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers import Dense
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.optimizers import Adadelta
from keras.optimizers import Adam
from keras.optimizers import Adamax
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

np.random.seed(7)


# train image data read and conversion to ndarray of dimensions (no. of images, width, height, channels)

def create_pd_input(path):
    with open(path, 'r') as train_file:
        image_files_names = train_file.readlines()
    for i in range(0, len(image_files_names)):
        image_files_names[i] = image_files_names[i].rstrip('\n')

    pixel_data = []
    for name in image_files_names:
        pixel_data.append(imread('/home/ubuntu/image_data/' + name + '.png'))
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

    # one hot encoding
    labels_ohe = np_utils.to_categorical(labels, nb_classes=10)
    return labels_ohe


X_pd_train = create_pd_input('/home/ubuntu/X_train.txt')
X_pd_test = create_pd_input('/home/ubuntu/X_test.txt')

label_train_ohe = process_labels('/home/ubuntu/y_train.txt')


def create_model():
    cnn_model = Sequential()

    cnn_model.add(Convolution2D(64, 7, 7, dim_ordering='tf', border_mode='same', input_shape=(32, 32, 3),  subsample=(2,2),W_regularizer='l2',W_constraint=maxnorm(3), bias=True))
    cnn_model.add((PReLU()))
    cnn_model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    #pool-1
    cnn_model.add(MaxPooling2D(pool_size=(3, 3),strides=(3,3)))
    cnn_model.add((PReLU()))
   
    #Conv2
    cnn_model.add(Convolution2D(128, 2, 2, border_mode='same',W_regularizer='l2', W_constraint=maxnorm(3), bias=True))
    cnn_model.add((PReLU()))
    cnn_model.add(Convolution2D(128, 2, 2,  border_mode='same',W_regularizer='l2', W_constraint=maxnorm(3), bias=True))
    cnn_model.add((PReLU()))
    cnn_model.add(Convolution2D(128, 2, 2, border_mode='same',W_regularizer='l2', W_constraint=maxnorm(3), bias=True))
    cnn_model.add((PReLU()))
    cnn_model.add(Convolution2D(128, 2, 2, border_mode='same',W_regularizer='l2', W_constraint=maxnorm(3), bias=True))
    cnn_model.add((PReLU()))
    cnn_model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    
    #Pool-2
    cnn_model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    
    #Conv2
    cnn_model.add(Convolution2D(256, 2, 2, border_mode='same', W_constraint=maxnorm(3), bias=True))
    cnn_model.add((PReLU()))
    cnn_model.add(Convolution2D(256, 2, 2,  border_mode='same', W_constraint=maxnorm(3), bias=True))
    cnn_model.add((PReLU()))
    cnn_model.add(Convolution2D(256, 2, 2,  border_mode='same', W_constraint=maxnorm(3), bias=True))
    cnn_model.add((PReLU()))
    cnn_model.add(Convolution2D(256, 2, 2, border_mode='same', W_constraint=maxnorm(3), bias=True))
    cnn_model.add((PReLU()))
    cnn_model.add(Convolution2D(256, 2, 2,  border_mode='same', W_constraint=maxnorm(3), bias=True))
    cnn_model.add((PReLU()))
    cnn_model.add(Convolution2D(256, 2, 2, border_mode='same', W_constraint=maxnorm(3), bias=True))
    cnn_model.add((PReLU()))

    cnn_model.add(Flatten())
    #Fully Connected Layer1
    cnn_model.add(Dense(4096, W_constraint=maxnorm(3)))
    cnn_model.add((PReLU()))
    #Fully Connected Layer2
    cnn_model.add(Dense(4096, W_constraint=maxnorm(3)))
    cnn_model.add((PReLU()))
    #Fully Connected Layer3
    cnn_model.add(Dense(10, activation='softmax'))
    ad=Adagrad(lr=0.01, epsilon=1e-05, decay=0.0)
  

    cnn_model.compile(loss='categorical_crossentropy', optimizer=ad, metrics=['accuracy'])
    return cnn_model

def getLabels(labels):
    f = open("Output.txt",'w')
    for l in labels:
        x=str(l)
        f.write(x+"\n")
    f.close()


def main():
    model = create_model()
    #Image data augmentation 
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
	vertical_flip=False) # randomly flip images
    datagen.fit(X_pd_train)
    model.fit_generator(datagen.flow(X_pd_train,label_train_ohe,
                        batch_size=1096),
                        samples_per_epoch=X_pd_train.shape[0],
                        nb_epoch=2000)
    pri = model.predict_classes(X_pd_test, batch_size=64, verbose=2)

    getLabels(pri)


if __name__ == '__main__':
main()
