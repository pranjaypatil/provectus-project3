
import os

import numpy as np
import tensorflow as tf
from PIL import Image


tf.app.flags.DEFINE_string('directory', '/home/ubuntu','''Directory to save tf recordfile''')

FLAGS = tf.app.flags.FLAGS
# Parameters
num_classes = 10
IMAGE_SIZE = 32
IMAGE_SHAPE = [IMAGE_SIZE, IMAGE_SIZE, 3]

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(images,name):
  """Method to convert image into TF Record Format"""
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]
  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing test images', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(10000):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())



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
    train_images=read_images_from("/home/ubuntu/X_test.txt","/home/dharamendra/image_data/")

    convert_to(train_images, 'testlarge')
    print ('Completed Successfully')
if __name__ == '__main__':
  main()