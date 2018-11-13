# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time

import tensorflow.python.platform
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Bathec size: must devide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir','D:\\Users\\Madhouse\\tensorflow\\data\\mnist','Directory to the input training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')

def weigth_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
  
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def inferenceCNN(images, keep_prob):
  W_conv1 = weigth_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])

  x_image = tf.reshape(images, [-1, 28, 28, 1])

  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weigth_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  W_fc1 = weigth_variable([7*7*64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weigth_variable([1024, NUM_CLASSES])
  b_fc2 = bias_variable([NUM_CLASSES])
  y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
  return y_conv

def lossCNN(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, float - [batch_size, NUM_CLASSES].

  Returns:
    loss: Loss tensor of type float.
  """
  loss = -tf.reduce_sum(labels*tf.log(logits))
  return loss
 
def trainingCNN(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar(loss.op.name, loss)
  # Create the gradient descent optimizer with the given learning rate.
  #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op  
  
def evaluationCNN(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size, NUM_CLASSES], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  #correct = tf.nn.in_top_k(logits, labels, 1)
  correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, NUM_CLASSES))
  keep_prob = tf.placeholder(tf.float32)
  return images_placeholder, labels_placeholder, keep_prob

def fill_feed_dictCNN(data_set, type, images_pl, labels_pl, keep_prob):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size ` examples.
  batch = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
  kp = 0.5
  if type.find("train") == -1:
    kp = 1.0 
  feed_dict = {
      images_pl: batch[0],
      labels_pl: batch[1],
	  keep_prob: kp
  }
  #print(' type =%s, keep_prob = %.1f'%(type,kp))
  return feed_dict

def do_evalCNN(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
			keep_prob,
            data_set,
            type):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dictCNN(data_set, type, images_placeholder,
                               labels_placeholder, keep_prob)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count / num_examples * 100.0
  print('  Num examples: %d  Num correct: %d  Precision: %g%%' %
        (num_examples, true_count, precision))
		
def run_training():
  st = time.time()
  print('start to training, time is %d'%(st))
  """Train MNIST for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data, one_hot=True)
  #print(' data_set =%s, train = %s, test= %s, validation = %s'%(str(data_sets),str(data_sets.train),str(data_sets.test),str(data_sets.validation)))
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder, keep_prob = placeholder_inputs(
        FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits = inferenceCNN(images_placeholder, keep_prob)

    # Add to the Graph the Ops for loss calculation.
    loss = lossCNN(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = trainingCNN(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = evaluationCNN(logits, labels_placeholder)

    # Build the summary operation based on the TF collection of Summaries.
    #summary_op = tf.merge_all_summaries()
    summary_op = tf.summary.merge_all()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
    #                                       graph_def=sess.graph_def)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph_def=sess.graph_def)

    # And then after everything is built, start the training loop.
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = fill_feed_dictCNN(data_sets.train, 'train', images_placeholder,
                                 labels_placeholder, keep_prob)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 200 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        saver.save(sess, FLAGS.train_dir, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_evalCNN(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
				keep_prob,
                data_sets.train, 'train')
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        do_evalCNN(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
				keep_prob,
                data_sets.validation, 'validation')
        # Evaluate against the test set.
        print('Test Data Eval:')
        do_evalCNN(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
				keep_prob,
                data_sets.test, 'test')
  tc = time.time() - st
  print('training end, time cost %.3f sec'%(tc))
				
def main(_):
  run_training()
  
if __name__ == '__main__':
  tf.app.run()

