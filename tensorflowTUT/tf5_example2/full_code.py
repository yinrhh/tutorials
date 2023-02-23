# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

# 定义误差，并建立优化器，使用优化器在每一次训练中减少误差
loss = tf.reduce_mean(tf.square(y-y_data))
# 神经网络中有多种优化器可供选择，0.5是学习效率，学习效率太大，梯度下降的时候会纠结。
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
### create tensorflow structure end ###

sess = tf.Session()
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    # 新版初始化语句
    init = tf.global_variables_initializer()
# 激活init，非常重要
sess.run(init)

for step in range(201):
    # 开始训练
    sess.run(train)
    if step % 20 == 0:
        # 每隔20步，打印出每次训练后的参数Weights（权重）、biases（偏置）
        # weight、biases是张量，无法直接输出，只能通过sess.run输出
        print(step, sess.run(Weights), sess.run(biases))


