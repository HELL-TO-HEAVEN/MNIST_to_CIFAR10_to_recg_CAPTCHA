# coding: UTF-8
from tensorflow.models.tutorials.image.cifar10 import cifar10
from tensorflow.models.tutorials.image.cifar10 import cifar10_input
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

FLAGS.data_dir = 'cifar10_data/'

cifar10.maybe_download_and_extract()