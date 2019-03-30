# coding: UTF-8
import tensorflow as tf
from tensorflow.models.tutorials.image.cifar10 import cifar10
from tensorflow.models.tutorials.image.cifar10 import cifar10_input
import tools
import numpy as np
#from datetime import datetime
import math,time
from datetime import datetime


STEP = 3000
batch_size = 128
data_dir = './cifar-10-batches-bin'
log_dir = './log'

image_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)   # ??
image_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

"""
定义输入输出层
"""
with tf.name_scope('input'):
    image_holder = tf.placeholder(tf.float32, shape=[batch_size, 24, 24, 3])   # 取的是图片中间的24X24的区域
    label_holder = tf.placeholder(tf.int32, shape=[batch_size])    # cifar10的label没有用one-hot，而是一维向量，每个数值代表每张图的类别
    # 注意label_holder的类型是整型，否则下面的in_top_k函数会报错
    # “TypeError: Value passed to parameter 'targets' has DataType float32 not in list of allowed values: int32, int64”

"""
第一层：
"""
# 卷积层conv1
with tf.name_scope('conv1'):
    with tf.name_scope('kernel1'):
        kernel1 = tools.Weight_with_WeightLoss(shape=[5, 5, 3, 64], stddev=5e-2, lamda=0)
        tools.variables_summaries(kernel1, 'conv1/kernel1')
    with tf.name_scope('conv_op'):
        conv1_result = tf.nn.conv2d(image_holder, kernel1, strides=[1, 1, 1, 1], padding='SAME')
    with tf.name_scope('bias_op'):
        b1 = tools.bias(0.1, shape=[64])
        tools.variables_summaries(b1, 'conv1/b1')
    with tf.name_scope('activate'):
        act1_result = tf.nn.relu(tf.nn.bias_add(conv1_result, b1))

# 池化层与局部响应归一化
with tf.name_scope('max_pool_and_norm1'):
    with tf.name_scope('max_pool'):
        pool1_result = tf.nn.max_pool(act1_result, ksize=[1, 2, 2, 1], strides=[1, 3, 3, 1], padding='SAME')
    with tf.name_scope('lrn1'):
        norm1_result = tf.nn.lrn(pool1_result, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)    # ??局部响应归一化


"""
第二层
"""
# 卷积层conv2
with tf.name_scope('conv2'):
    with tf.name_scope('kernel2'):
        kernel2 = tools.Weight_with_WeightLoss(shape=[5, 5, 64, 64], stddev=5e-2, lamda=0)
        tools.variables_summaries(kernel2, 'conv2/kernel2')
    with tf.name_scope('conv_op'):
        conv2_result = tf.nn.conv2d(norm1_result, kernel2, strides=[1, 1, 1, 1], padding='SAME')
    with tf.name_scope('bias_op'):
        b2 = tools.bias(0.1, shape=[64])
        tools.variables_summaries(b2, 'conv2/b2')
    with tf.name_scope('activate'):
        act2_result = tf.nn.relu(tf.nn.bias_add(conv2_result, b2))

# 池化层与局部响应归一化
with tf.name_scope('max_pool_and_norm2'):
    with tf.name_scope('max_pool'):
        pool2_result = tf.nn.max_pool(act2_result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.name_scope('lrn2'):
        norm2_result = tf.nn.lrn(pool2_result, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)


"""
第三层
"""
# 全连接层fulc1
with tf.name_scope('fulc1'):
    with tf.name_scope('reshape'):
        fulc1_in = tf.reshape(norm2_result, [batch_size, -1])
        col_dim = fulc1_in.get_shape()[1].value               # 牛逼？？
    with tf.name_scope('ful_w1'):
        ful_w1 = tools.Weight_with_WeightLoss(shape=[col_dim, 384], stddev=0.04, lamda=0.004)
        tools.variables_summaries(ful_w1, 'fulc1/ful_w1')
    with tf.name_scope('bias_op'):
        b3 = tools.bias(0.1, shape=[384])
        tools.variables_summaries(b3, 'fulc1/b3')
    fulc1_result = tf.nn.relu(tf.matmul(fulc1_in, ful_w1) + b3)


"""
第四层
"""
# 全连接层fulc2
with tf.name_scope('fulc2'):
    with tf.name_scope('ful_w2'):
        ful_w2 = tools.Weight_with_WeightLoss(shape=[384, 192], stddev=0.04, lamda=0.004)
        tools.variables_summaries(ful_w2, 'fulc2/ful_w2')
    with tf.name_scope('bias'):
        b4 = tools.bias(0.1, shape=[192])
        tools.variables_summaries(b4, 'fulc2/b4')
    fulc2_result = tf.nn.relu(tf.matmul(fulc1_result, ful_w2) + b4)


# 全连接层fulc3
with tf.name_scope('fulc3'):
    with tf.name_scope('ful_w3'):
        ful_w3 = tools.Weight_with_WeightLoss(shape=[192, 10], stddev=1/192, lamda=0.0)
        tools.variables_summaries(ful_w3, 'fulc3/ful_w3')
    with tf.name_scope('bias'):
        b5 = tools.bias(0.1, shape=[10])
        tools.variables_summaries(b5, 'fulc3/b5')
    output = tf.matmul(fulc2_result, ful_w3) + b5


"""
损失
"""
with tf.name_scope('loss'):
    loss_total = tools.LOSS(output, label_holder)
    loss_summary = tf.summary.scalar('losses', loss_total)

"""
优化
"""
with tf.name_scope('train_step'):
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss_total)


"""
精度
"""
with tf.name_scope('accuracy'):
    # 返回一个布尔向量，第一个参数是计算出的prediction,第二个是实际的label,第三个是ｋ表示前ｋ个最大的数，这里取１
    # （跟tf.equal(tf.argmax(x,1),label)）是一样的
    top_k_op = tf.nn.in_top_k(output, label_holder, 1)


"""
开启会话训练
"""
with tf.Session() as sess:
    start_count = datetime.now()

    init = tf.initialize_all_variables()
    sess.run(init)

    # 合并所有的summary
    merge_summary = tf.summary.merge_all()
    # 将日志写入log_dir
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)

    # 数据集的读取需要打开线程
    tf.train.start_queue_runners()

    for i in range(1, STEP):
        start = time.time()

        image_train_batch, label_train_batch = sess.run([image_train, labels_train])   # 分开写俩sess.run也行
        # 其中batch_size已经作为参数嵌入image_train和labels_train

        train_op, loss, summary = sess.run([train_step, loss_total, merge_summary],
                                           feed_dict={image_holder: image_train_batch, label_holder: label_train_batch})
        # 每次都要进行日志记录
        train_writer.add_summary(summary, global_step=i)

        duration = time.time() - start



        if i % 50 == 0:
            # 训练一个batch所需时间
            sec_per_batch = float(duration)
            # 训练精度
            y_train = sess.run(top_k_op, feed_dict={image_holder: image_train_batch, label_holder: label_train_batch})
            print('After %d step, loss = %g, train accuracy = %g  ; (%g sec/batch)'
                  % (i, loss, np.sum(y_train)/batch_size, sec_per_batch))

    # 测试集精度
    num_test = 10000      # 测试集样本数量
    num_iter = int(math.ceil(num_test/batch_size))           # math的ceil方法向上取整; 分批进行，这是总批数
    total_sample_count = num_iter * batch_size
    true_num = 0       # 分类正确的个数
    step = 0                   # 分批进行，这是批次

    # 一批一批计算正确分类个数
    while step < num_iter:
        image_test_batch, label_test_batch = sess.run([image_test, labels_test])     # ？？？
        predictions = sess.run(top_k_op,
                               feed_dict={image_holder: image_test_batch, label_holder: label_test_batch})
        true_num += np.sum(predictions)     # 不是布尔值了？？？直接可以数值运算了？？？
        step += 1
    print('The accuracy in test_data = ', true_num/total_sample_count)

    end_count = datetime.now()
    print('Spent time: ', end_count-start_count)