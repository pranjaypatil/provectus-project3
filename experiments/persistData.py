import numpy as np
import cv2
from scipy.misc import imread

from keras.utils import np_utils

np.random.seed(7)

# Process Train Data
def create_pd_input(path):
    with open(path, 'r') as train_file:
        image_files_names = train_file.readlines()
    for i in range(0, len(image_files_names)):
        image_files_names[i] = image_files_names[i].rstrip('\n')

    pixel_data = []
    for name in image_files_names:
        pixel_data.append(imread('/home/ubuntu/image_data/' + name + '.png', mode='RGB'))
        if counter>= 10000:
                writeData=[]
                writeData= np.asarray(pixel_data)
                writeData.astype('float32') / 255.0
                np.save("/home/ubuntu/file"+str(i),writeData,True,True)
                counter=0
                i=i+1

# Process Test Data
def create_pd_testInput(path):
    with open(path, 'r') as train_file:
        image_files_names = train_file.readlines()
    for i in range(0, len(image_files_names)):
        image_files_names[i] = image_files_names[i].rstrip('\n')

    pixel_data = []
    for name in image_files_names:
        pixel_data.append(imread('/home/ubuntu/image_data/' + name + '.png', mode='RGB'))

    pixel_data = np.asarray(pixel_data)
    np.save("/home/ubuntu/testDataFile",pixel_data,True,True)

# Reading Labels and applying one-hot encoding
def process_labels(path)
 
    with open(path, 'r') as label_file:
        labels = label_file.readlines()
    for i in range(0, len(labels)):
        labels[i] = labels[i].rstrip('\n')

    # one hot encoding
    labels_ohe = np_utils.to_categorical(labels, nb_classes=10)
    return labels_ohe


def main():

    create_pd_input('/home/ubuntu/X_train.txt')
    create_pd_testInput('/home/ubuntu/X_test.txt')

    label_train_ohe = process_labels('/home/ubuntu/y_train.txt')
    
    for(i in range(0,5)):
        data = np.load("/home/ubuntu/file"+str(i)+".npy",None,True,True,'ASCII')
        X_pd_train.append(data)

    X_pd_train= np.load("/home/ubuntu/testDataFile.npy",None,True,True,'ASCII')  

if __name__ == '__main__':
    main()