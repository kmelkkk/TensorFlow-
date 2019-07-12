import numpy as np
import tensorflow as tf

x_data = np.random.rand(100)
y_data = x_data * 10 + 5

#构造线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y=k*x_data+b


#二次代价函数   square求平方
loss= tf.reduce_mean(tf.square(y_data-y))

#定义一个梯度下降法来进行训练的优化器

optimizer=tf.train.GradientDescentOptimizer(.2)

train=optimizer.minimize(loss)

init=tf.global_variables_initializer()

with tf.Session() as ss:
    ss.run(init)
    for step in range(201):
        ss.run(train)
        if step %10==0:
            print(step,ss.run([k,b]))
