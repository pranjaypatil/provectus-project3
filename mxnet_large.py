import sys
import math
import mxnet as mx
import numpy as np
import logging
from scipy.misc import imread

logger_here = logging.getLogger()
logger_here.setLevel(logging.DEBUG)

class mxnet_cifar10(object):

    def __init__(self):
        pass
    
    def create_pd_input(self, path):

        '''
        Read input image and convert to numpy array.
        
        Input parameter(s):
        - path: The path of file containing filenames of images

        return:
        - An numpy array (ndarray) with pixel data for all images
        shape (samples, width, height, channels)
        '''

        with open(path, 'r') as train_file:
            image_files_names = train_file.readlines()
        for i in range(0, len(image_files_names)):
            image_files_names[i] = image_files_names[i].rstrip('\n')

        pixel_data = []
        for name in image_files_names:
            pixel_data.append(imread('./images/' + name + '.png'))
        pixel_data = np.asarray(pixel_data)
        return pixel_data.astype('float32') / 255.0

    def process_labels(self, path):
        
        '''
        Read the input path for label file and return numpy array with labels

        Input Parameters:
        - path: The path to the file that contains labels

        returns:
        - The numpy array with labels shape:(samples, 1)
        '''

        with open(path, 'r') as label_file:
            labels = label_file.readlines()
        for i in range(0, len(labels)):
            labels[i] = labels[i].rstrip('\n')

        return np.asarray(labels).astype('float32')

    def getLabels(self, predictions):

        '''
        Writes the predictions to file named output.txt

        Input parameter:
        - predictions: The predictions in numpy array with shape (samples, number of classes)

        - returns: No return
        '''

        f = open("output.txt",'w')
        for probs in predictions:
            probs = probs.tolist()
            label = max(probs)
            f.write(str(probs.index(label)) + '\n')
        f.close()

def main():

    cifar10 = mxnet_cifar10()

    X_train = cifar10.create_pd_input(sys.argv[2])
    X_test = cifar10.create_pd_input(sys.argv[3])

    y_train = cifar10.process_labels(sys.argv[4])

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
    
    # Optimizers
    sgd = mx.optimizer.SGD(learning_rate=0.005, momentum=0.9, wd=0.00001)
    ada = mx.optimizer.AdaGrad(learning_rate=0.01, eps=1e-8, wd=0.00001)
    
    #Initializers
    normal = mx.initializer.Normal(sigma=0.01)
    uniform = mx.initializer.Uniform(scale=0.07)
    
    #Loss Metric
    cross_entropy = mx.metric.CrossEntropy()

    num_gpus = int(sys.argv[1])

    #The model
    model = mx.model.FeedForward(
      symbol = softmax,  
      ctx = [mx.gpu(i) for i in range(0, num_gpus)],
      initializer = normal,
      num_epoch = 100,
      optimizer = ada,
      eval_metric = mx.metric.CrossEntropy())
    #ctx = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)],

    model.fit(X=training_iter, eval_metric = cross_entropy, batch_end_callback=mx.callback.Speedometer(128))

    predictions = model.predict(X=X_test)
    
    cifar10.getLabels(predictions)

if __name__ == '__main__':
    main()