# Image Classification on CIFAR 10

##Overview

Image classification was a challenging project of Microsoft CIFAR10 where we were asked to classify the images on the basis of 10 given classes.We applied Deep Learning using Tensor Flow and Keras.
The scripts used for this classification are described beow:

- keras_final.py        : Reads and preprocessing data, model training and predicting testing data.
- images_to_tfrecord.py : create tensor flow record format file from image data.
- multi_gpu.py          : Reads tensor flow record format file, image pre-processing and multi CPU training model.
- mxnet_large.py        : Improved network and Image preprocessing than mxnet_small.py. CNN based on mxnet API for  GPU based training.

##Problem Decsrition

The problem was to identify the category of large collecton of small images. Each image is a 32x32 pixel with three color channels: standard RGB format. There were 60,000 images splitting as 50,000 images for training the model and 10,000 for testing. The whole dataset was around 300 MB. The class/category for an image to be classified were:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

##Installations

Run init.sh for all required installations. This file installs the following:

- Numpy
- Keras
- TensorFlow version 0.10  and 0.11
- Cuda 7.5 tool kit
- Cudnn library 5.1 for Cuda 7.5

