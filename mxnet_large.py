import math
import mxnet as mx
import numpy as np
import logging
from scipy.misc import imread

logger_here = logging.getLogger()
logger_here.setLevel(logging.DEBUG)

# train image data read and conversion to ndarray of dimensions (no. of images, width, height, channels)

def create_pd_input(path):
    with open(path, 'r') as train_file:
        image_files_names = train_file.readlines()
    for i in range(0, len(image_files_names)):
        image_files_names[i] = image_files_names[i].rstrip('\n')

    pixel_data = []
    for name in image_files_names:
        pixel_data.append(imread('./images/' + name + '.png'))
    pixel_data = np.asarray(pixel_data)
    return pixel_data.astype('float32')

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
    return np.asarray(labels).astype('float32')

X_train = create_pd_input('./metadata/X_train.txt')
X_test = create_pd_input('./metadata/X_test.txt')

y_train = process_labels('./metadata/y_train.txt')
#y_test = process_labels('./metadata/y_small_test.txt')

X_train -= np.mean(X_train)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test)
X_test /= np.std(X_test, axis=0)

# Iterator over the data

training_iter = mx.io.NDArrayIter(X_train, y_train, batch_size=128)
#test_iter = mx.io.NDArrayIter(X_test, batch_size=32)

# Here be neural network

data = mx.symbol.Variable('data')

#first convolutional layer
conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias = False, num_filter=32, name='conv1')

#Activaition for conv1 layerl
act1 = mx.symbol.Activation(data=conv1, act_type='relu', name='act1')

#second convolutional layer
conv2 = mx.symbol.Convolution(data=act1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias = False, num_filter=32, name='conv2')

#Activaition for conv2 layer
act2 = mx.symbol.Activation(data=conv2, act_type='relu', name='act2')

#mxpl1
mxpl1 = mx.symbol.Pooling(data=act2, kernel=(2,2), pool_type='max', name='mxpl1')

#drp1
drp1 = mx.symbol.Dropout(data=mxpl1, p=0.2, name='drp1')

#conv3
conv3 = mx.symbol.Convolution(data=drp1, kernel=(3, 3), stride=(1,1), pad=(1,1), no_bias=False, num_filter=64, name='conv3')

#act3
act3 = mx.symbol.Activation(data=conv3, act_type='relu', name='act3')

#mxpl2
mxpl2 = mx.symbol.Pooling(data=act3, kernel=(2,2), pool_type='max', name='mxpl2')

#Drp2
drp2 = mx.symbol.Dropout(data=mxpl2, p=0.2, name='drp2')

#conv4
conv4 = mx.symbol.Convolution(data=drp2, kernel=(3, 3), stride=(1,1), pad=(2,2), no_bias=False, num_filter=64, name='conv4')

#act4
act4 = mx.symbol.Activation(data=conv4, act_type='relu', name='act4')

#mxpl3
mxpl3 = mx.symbol.Pooling(data=act4, kernel=(2,2), pool_type='max', name='mxpl3')

#drp3
drp3 = mx.symbol.Dropout(data=mxpl3, p=0.2, name='drp3')

#Flattern the output
flt1 = mx.symbol.Flatten(data=drp3, name='flt1')

#Fully connected layer 1
ful1 = mx.symbol.FullyConnected(data=flt1, num_hidden=512, name='ful1')

#Activation for Fully connected 1
actf1 = mx.symbol.Activation(data=ful1, act_type='relu', name='actf1')

#drp5
drp5 = mx.symbol.Dropout(data=actf1, p=0.2, name='drp5')

#Fully connected layer 2
ful2 = mx.symbol.FullyConnected(data=drp5, num_hidden=64, name='ful2')

#Activation for Fully connected 2
actf2 = mx.symbol.Activation(data=ful2, act_type='relu', name='actf2')

#drp6
drp6 = mx.symbol.Dropout(data=actf2, p=0.5, name='drp6')

#Fully connected layer 3
ful3 = mx.symbol.FullyConnected(data=drp6, num_hidden=10, name='ful3')

#Activation for Fully connected 3
softmax = mx.symbol.SoftmaxOutput(data=ful3, name='softmax')

# NN ends here, phew!

ada = mx.optimizer.AdaGrad(learning_rate=0.01, eps=1e-08, wd=0.0)

model = mx.model.FeedForward(
  symbol=softmax,  
  ctx = [mx.gpu(0)],
  initializer=mx.initializer.Uniform(scale=0.07),
  num_epoch=100,
  optimizer=ada)
#ctx = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)],

#model.fit(X=X_train, y=y_train)
model.fit(X=training_iter, eval_metric='accuracy', batch_end_callback=mx.callback.Speedometer(128))

def getLabels(predictions):
    f = open("output.txt",'w')
    for probs in predictions:
        probs = probs.tolist()
        label = max(probs)
        f.write(str(probs.index(label)) + '\n')
    f.close()
predictions = model.predict(X=X_test)
getLabels(predictions)
