import mxnet as mx
import numpy as np
import graphviz
from scipy.misc import imread
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
    #labels_ohe = np_utils.to_categorical(labels, nb_classes=10)
    #return labels_ohe
    return np.asarray(labels)

X_pd_train = create_pd_input('./data/X_small_train.txt')
X_pd_test = create_pd_input('./data/X_small_test.txt')

label_train_ohe = process_labels('./data/y_small_train.txt')
label_test_ohe = process_labels('./data/y_small_test.txt')

label_train_ohe = label_train_ohe.astype('int')
label_test_ohe = label_test_ohe.astype('int')

# Here be neural network

# The data variable, its just a placeholder (or is it?)
data = mx.symbol.Variable('data')

#first convolutional layer
conv1 = mx.symbol.Convolution(data=data, 
                              kernel=(1, 1), 
                              stride=(1, 1),
                              no_bias = False,
                              num_filter=32, 
                              name='conv1')

#Activaition for conv1 layer
act1 = mx.symbol.Activation(data=conv1, act_type='relu', name='act1')

#Dropout layer
drp1 = mx.symbol.Dropout(data=act1, p=0.2, name='drp1')

#second convolutional layer
conv2 = mx.symbol.Convolution(data=drp1, 
                              kernel=(1, 1), 
                              stride=(1, 1), 
                              no_bias = False,
                              num_filter=32, 
                              name='conv2')

#Activaition for conv2 layer
act2 = mx.symbol.Activation(data=conv2, act_type='relu', name='act2')

#Maxpooling
mxpl1 = mx.symbol.Pooling(data=act2, kernel=(2, 2), pool_type='max', name='mxpl1')

#Flattern the output
flt1 = mx.symbol.Flatten(data=mxpl1, name='flt1')

#Fully connected layer 1
ful1 = mx.symbol.FullyConnected(data=flt1, num_hidden=512, name='ful1')

#Activaition for fully connected layer 1
act3 = mx.symbol.Activation(data=ful1, act_type='relu', name='act3')

#Dropout layer
drp2 = mx.symbol.Dropout(data=act3, p=0.5, name='drp2')

#Final Fully connected layer
ful2 = mx.symbol.FullyConnected(data=drp2, num_hidden=10, name='ful2')

#Activation for Fully connected 2
softmax = mx.symbol.SoftmaxOutput(data=ful2, name='softmax')

# Iterator over the data
training_iter = mx.io.NDArrayIter(X_pd_train, label_train_ohe, batch_size=10)
test_iter = mx.io.NDArrayIter(X_pd_test, label_test_ohe, batch_size=10)

def norm_stat(d):
    """The statistics you want to see.
    We compute the L2 norm here but you can change it to anything you like."""
    return mx.nd.norm(d)/np.sqrt(d.size)

model = mx.model.FeedForward(
    symbol=softmax,
    num_epoch=100,
    learning_rate=0.1,
    momentum=0.9,
    wd=0.00001)

mon = mx.mon.Monitor(
    10,
    norm_stat,
    sort=True)

model.fit(X=training_iter, 
          eval_data=test_iter,
          monitor=mon,
          batch_end_callback=mx.callback.Speedometer(10, 50))

print 'Accuracy:', model.score(X=test_iter, eval_metric='acc',num_batch=10, reset=True) * 100, '%'
