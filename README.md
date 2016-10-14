# Image Classification on CIFAR 10

##Overview

Image classification was a challenging project of Microsoft CIFAR10 where we were asked to classify the images on the basis of 10 given classes.We applied Deep Learning using Tensor Flow and Keras.
The scripts used for this classification are described beow:

- keras_final.py        : Reads and preprocessing data, model training and predicting testing data.
- images_to_tfrecord.py : Create tensor flow record format file from image data.
- multi_gpu.py          : Reads tensor flow record format file, image pre-processing and multi CPU training model.
- mxnet_large.py        : Convolutional Neural Network based on MXNet API, supports multiple GPUs.

##Problem Description

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

##How to Run
Before executing any python script, the following dependenices need to be installed.
- numpy
- pillow
- scipy
- keras
- mxnet (v. 0.7.0)
- tensorflow (v. 0.10 for keras and 0.11 for multi-gpu support)
- CUDA (v. 7.5) and cuDNN(v. 5.1)

__Execution Instructions__
- **mxnet_large.py** - command to run:- python mxnet_large.py #of_gpus path_to_train_file path_to_test_file path_to_train_labels
- __For Multi GPU using Tensorflow__
run all using python \<filename> in the following order
  - run images_to_tfrecord.py
  - run testimages_to_tfrecord.py
  - run multi_gpu_tensorflow.py
  - run multi_gpu_eval_tensorflow.py
- __Keras based classifiers__
  run all files using python \<filename>
  - keras_final.py and keras_without_aug.py

##Project Report
Please refer to the project3_report.pdf file for a detailed overview of layers in various implementation of CNNs using different frameworks.
