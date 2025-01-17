# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf

# tensorflow只能处理float32的形式
# placeholder与variable不同的是，placeholder在sess.run的时候才传入值
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    # placeholder是外部传入的值，而variable相当于参数
    # 官方推荐使用dataset替代feed_dict
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))
