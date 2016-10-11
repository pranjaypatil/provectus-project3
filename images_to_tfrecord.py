
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from keras.utils import np_utils

tf.app.flags.DEFINE_string('directory', '/home/dharamendra','''Directory to save tf recordfile''')
# tf.app.flags.DEFINE_integer('validation_size', 10000,
#                             'Number of examples to separate from the training '
#                             'ckpt for the validation set.')
FLAGS = tf.app.flags.FLAGS
# Parameters
num_classes = 10
IMAGE_SIZE = 32
IMAGE_SHAPE = [IMAGE_SIZE, IMAGE_SIZE, 3]


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(images, labels, name):
  num_examples = labels.shape[0]
  print('labels shape is ', labels.shape[0])
  if images.shape[0] != num_examples:
    raise ValueError("Images size %d does not match label size %d." %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]
  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())

def process_labels(path):
    '''
    Read the input path for label file and return the OHE for labels
    '''
    with open(path, 'r') as label_file:
        labels = label_file.readlines()
    for i in range(0, len(labels)):
        labels[i] = labels[i].rstrip('\n')


    return np.array(labels)

def read_images_from(path,directory):
  images = []
  with open(path, 'r') as train_file:
      image_files_names = train_file.readlines()
  for i in range(0, len(image_files_names)):
      image_files_names[i] = image_files_names[i].rstrip('\n')
  for filename in image_files_names:
    im = Image.open(directory+filename+".png")
    im = np.asarray(im, np.uint8)
    images.append([int(filename), im])

  images_only = [np.asarray(image[1], np.uint8) for image in images]
  images_only = np.array(images_only)

  print(images_only.shape)
  return images_only
def main():
    train_images=read_images_from("/home/dharamendra/X_small_train.txt","/home/dharamendra/image_data/")
    train_labels=process_labels("/home/dharamendra/y_small_train.txt")

    convert_to(train_images, train_labels, 'train')

if __name__ == '__main__':
  main()