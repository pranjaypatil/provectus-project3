from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
import multi_gpu_tensorflow
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=25000
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', '/home/ubuntu/cifar10_eval', """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test', """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/ubuntu/ckpt',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5, """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False, """Whether to run eval only once.""")
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.


def read_and_decode(filename_queue, imshape, flatten=False):
  """Method for reading TFRecordReader format file"""
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      'image_raw': tf.FixedLenFeature([], tf.string),
    })

  image = tf.decode_raw(features['image_raw'], tf.uint8)

  if flatten:
    num_elements = 1
    for i in imshape: num_elements = num_elements * i
    print (num_elements)
    image = tf.reshape(image, [num_elements])
    image.set_shape(num_elements)
  else:
    image = tf.reshape(image, imshape)
    image.set_shape(imshape)
    image = tf.cast(image, tf.float32)
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  return image


def distorted_inputs(image,imshape):
    """Image Preprocessing
    Args:
         image: from read_and_decode method
         imshape: for reshaping the images
    Returns:
        image:distorted images
    """
    # Reshape to imshape as distortion methods need this shape
    reshaped_image = tf.reshape(image, imshape)

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(reshaped_image)
    #Randomly brighten the images
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    #Randomly contrast the images
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(distorted_image)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(float_image,min_queue_examples,FLAGS.batch_size)

def _generate_image_and_label_batch(image, min_queue_examples,
                                    batch_size):
    """Method to create Queue Runner"""
    num_preprocess_threads = 16
    images = tf.train.batch(
        [image],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)
    return images


def eval_once(saver, prediction):
  """Run Eval once.
  Args:
    saver: Saver.
    prediction: Logits from model inference method
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    print ('CKPT Directory found')
    if ckpt and ckpt.model_checkpoint_path:
      # Restore model files from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      print('No checkpoint exist!!')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = []
    try:

      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(5000/ FLAGS.batch_size))

      step = 0
      preds=[]
      while step < num_iter and not coord.should_stop():
          p = sess.run(prediction)
          preds.append(np.argmax(p, 1))
          step = step + 1
          print(len(preds))
      pred = np.concatenate(preds)
      np.savetxt('OutputLabels.txt', pred, fmt='%.0f')


    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads)

def evaluate():
  """Evaluate for a number of steps."""
  print ('In Evaluate')
  with tf.Graph().as_default() as g:
    # Get testing images.
    filename_queue = tf.train.string_input_producer(['/home/ubuntu/testlarge.tfrecords'])
    images=read_and_decode(filename_queue,imshape=[32, 32, 3])
    images= distorted_inputs(images,imshape=[32, 32, 3])

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = multi_gpu_tensorflow.inference(images)

    # Calculate predictions.
    prediction = tf.nn.softmax(logits)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    while True:
        eval_once(saver, prediction)
        if FLAGS.run_once:
            break
        time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):
  evaluate()


if __name__ == '__main__':
  main()