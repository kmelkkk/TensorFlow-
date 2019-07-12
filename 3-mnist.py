import tensorflow as  tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples// batch_size

# 定义两个占位
# x有28*28个列，y是标签一共是0-9 10个可能
x = tf.placeholder(tf.float32, [None, 784])

y = tf.placeholder(tf.float32, [None, 10])
# 创建简单的神经网络
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, w) + b)

# 二次代价

loss = tf.reduce_mean(tf.square(y - prediction))

train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

# tf.argmax(input,axis)根据axis取值的不同返回每行或者每列最大值的索引,返回一维张量最大值的位置
# 　axis=0时比较每一列的元素，将每一列最大元素所在的索引记录下来，输出每一列最大元素所在的索引数组。1是列

correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print('Iter' + str(epoch) + "  准确率：" + str(acc))
