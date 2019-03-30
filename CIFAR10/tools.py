# coding: UTF-8
import tensorflow as tf


def Weight_with_WeightLoss(shape, stddev, lamda):
    w = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))

    if lamda is not None:
        WeightLoss = tf.multiply(tf.nn.l2_loss(w), lamda, name='WeighLoss')
        tf.add_to_collection('losses', WeightLoss)

    return w


def bias(number, shape):
    b = tf.constant(number, shape=shape, name='bias')
    return b


def LOSS(input, labels):

    label2num = tf.cast(labels, tf.int64)     # why??
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=input, labels=label2num, name='total_loss')

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')     # 为啥要加这一步？？

    #total_loss = tf.add_to_collection('losses', cross_entropy_mean)
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def variables_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/'+name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('stddev/'+name, stddev)
        tf.summary.scalar('max/'+name, tf.reduce_max(var))
        tf.summary.scalar('min/'+name, tf.reduce_min(var))
        tf.summary.histogram(name, var)
        #tf.summary.histogram()

