import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# 从-0.5到0.5 200个数字，一列
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 定义两个占位符
# placeholder(dtype, shape=None, name=None)
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))
xinhao = tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.tanh(xinhao)

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
xinhao_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(xinhao_L2)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as  ss:
    ss.run(tf.global_variables_initializer())

    for _ in range(2000):
        ss.run(train, feed_dict={x: x_data, y: y_data})

        # 获得预测值
        prediction_value = ss.run(prediction, feed_dict={x: x_data})

        # 画图
        plt.figure()
        plt.scatter(x_data, y_data)
        plt.plot(x_data, prediction_value, 'r', lw=5)
        plt.show()
