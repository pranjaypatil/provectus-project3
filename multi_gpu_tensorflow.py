import os.path
import time
from datetime import datetime
import math
import numpy as np
import tensorflow as tf
from six.moves import xrange
from datetime import datetime
from numpy.random import shuffle
import os.path
import re
import time
# Parameters
display_step = 1
val_step = 5
save_step = 50
IMAGE_PIXELS = 32 * 32 * 3
NEW_LINE = '\n'
pixels=32*32

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('train_dir', '/home/ubuntu/ckpt',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('num_epochs', 50000, 'Number of epochs to run trainer.')
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 4,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', True, """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('batch_size', 100, """Batch Size""")
tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")

# Constants describing the training process.
TOWER_NAME = 'tower'
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
IMAGE_SIZE = 32
NUM_EPOCHS_PER_DECAY_1=200
NUM_EPOCHS_PER_DECAY_2=250
NUM_EPOCHS_PER_DECAY_3=300
LEARNING_RATE_DECAY_FACTOR=0.1
num_batches_per_epoch=50000/100
decay_steps_1=int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY_1)
decay_steps_2=int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY_2)
decay_steps_3=int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY_3)
INITIAL_LEARNING_RATE=0.001

def read_and_decode(filename_queue, imshape, flatten=False):

  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      'image_raw': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64)
    })

  image = tf.decode_raw(features['image_raw'], tf.uint8)

  if flatten:
    num_elements = 1
    for i in imshape: num_elements = num_elements * i
    print num_elements
    image = tf.reshape(image, [num_elements])
    image.set_shape(num_elements)
  else:
    image = tf.reshape(image, imshape)
    image.set_shape(imshape)


    image = tf.cast(image, tf.float32)
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['label'], tf.int32)

  return image, label


def _random_brightness_helper(image):
    return tf.image.random_brightness(image, max_delta=63)


def _random_contrast_helper(image):
    return tf.image.random_contrast(image, lower=0.2, upper=1.8)


def distorted_inputs(filename, batch_size, num_epochs, num_threads,
                     imshape, num_examples_per_epoch=25000, flatten=True):

    if not num_epochs:
        num_epochs = None

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs, name='string_DISTORTED_input_producer')

        # Even when reading in multiple threads, share the filename
        # queue.
        image, label = read_and_decode(filename_queue, imshape)

        # Reshape to imshape as distortion methods need this shape
        image = tf.reshape(image, imshape)

        # Image processing for training the network. Note the many random
        # distortions applied to the image.

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(image)
        #
        # Randomly apply image transformations in random_functions list
        random_functions = [_random_brightness_helper, _random_contrast_helper]
        shuffle(random_functions)
        for fcn in random_functions:
            distorted_image = fcn(distorted_image)

        # # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_whitening(distorted_image)

        if flatten:
            num_elements = 1
            for i in imshape: num_elements = num_elements * i
            image = tf.reshape(float_image, [num_elements])
        else:
            image = float_image
        print ("image flatten shape",image.get_shape())
        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)
        images, sparse_labels = tf.train.shuffle_batch([image, label],
                                                       batch_size=batch_size,
                                                       num_threads=num_threads,
                                                       capacity=min_queue_examples + 3 * batch_size,
                                                       enqueue_many=False,
                                                       min_after_dequeue=min_queue_examples,
                                                       name='batching_shuffling_distortion')

    return images, sparse_labels





def _variable_on_cpu(name, shape, initializer):
    """
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Create an initialized Variable with weight decay.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. .
    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.contrib.layers.xavier_initializer()) #Xavier Intialization for Weight Variables
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inference(images):
    """Build the model.
    Args:
      images: Images returned from distorted_inputs() or inputs().
    Returns:
      Logits.
    """
    # conv1
    # Reshape input picture
    print('In Inference ', images.get_shape(), type(images))
    images = tf.reshape(images, shape=[-1, 32, 32, 3])
    print(images.get_shape())
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 3, 64], stddev=1./math.sqrt(pixels), wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64], stddev=1./math.sqrt(pixels), wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)


    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64], stddev=1./math.sqrt(pixels), wd=0.0)
        conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)

    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 128], stddev=1./math.sqrt(pixels), wd=0.0)
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope.name)

        # pool2
        pool4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
        # norm2
        norm4 = tf.nn.lrn(pool4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm4')

    # conv2
    with tf.variable_scope('conv5') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=1./math.sqrt(pixels), wd=0.0)
        conv = tf.nn.conv2d(norm4, kernel, [1, 2, 2, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope.name)

        # norm3
        norm5 = tf.nn.lrn(conv5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm5')
        # pool3
        pool5 = tf.nn.max_pool(norm5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
        shape = pool5.get_shape().as_list()
        pool5 = tf.reshape(pool5, [FLAGS.batch_size, shape[1] * shape[2] * shape[3]])

    # local6
    with tf.variable_scope('local6') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool5, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)


    # local7
    with tf.variable_scope('local7') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, 10], stddev=1 / 192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [10], tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)


    return softmax_linear


def loss(logits, labels_batch):
    """Add L2 loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs
    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    print (logits.get_shape())

    num_labels = 10 #Number of Categories

    sparse_labels = tf.reshape(labels_batch, [-1, 1])
    derived_size = tf.shape(labels_batch)[0]
    indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    outshape = tf.pack([derived_size, num_labels])
    labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
    print (labels.get_shape())
    labels=tf.reshape(labels,[FLAGS.batch_size,10])
    print(labels.get_shape())
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def tower_loss(scope):
    """Calculate the total loss on a single tower running the CIFAR model.
    Args:
      scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """
    # Get images and labels for CIFAR-10.
    images, labels = distorted_inputs(filename='/home/ubuntu/largetrain.tfrecords',
                                                       batch_size=FLAGS.batch_size,
                                                       num_epochs=FLAGS.num_epochs,
                                                      num_threads=16, imshape=[32, 32, 3])

    # Build inference Graph.
    logits = inference(images)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = loss(logits, labels)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        # Calculate the learning rate schedule.
        num_batches_per_epoch = (50000 / FLAGS.batch_size)
        decayed_learning_rate_1 = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                                             global_step,
                                                             decay_steps_1,
                                                             LEARNING_RATE_DECAY_FACTOR,
                                                             staircase=True)
        decayed_learning_rate_2 = tf.train.exponential_decay(decayed_learning_rate_1,
                                                             global_step,
                                                             decay_steps_2,
                                                             LEARNING_RATE_DECAY_FACTOR,
                                                             staircase=True)
        lr = tf.train.exponential_decay(decayed_learning_rate_2,
                                                           global_step,
                                                           decay_steps_3,
                                                           LEARNING_RATE_DECAY_FACTOR,
                                                           staircase=True)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.GradientDescentOptimizer(lr)

        tower_grads = []
        for i in xrange(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):#Multi GPU Code
                with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                    print('Working on: %s_%d' % (TOWER_NAME, i))
                    # Calculate the loss for one tower of the model but it is shared across the towers
                    loss = tower_loss(scope)

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Calculate the gradients for the batch of data on this CIFAR tower.
                    grads = opt.compute_gradients(loss)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)

        # Calculate the mean of each gradient.
        grads = average_gradients(tower_grads)

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build an initialization operation to run below.
        init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

        # Running operations on the Graph.
        sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        threads = tf.train.start_queue_runners(sess=sess)
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / float(duration)
                sec_per_batch = float(duration) / FLAGS.num_gpus

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch))
            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    main()
