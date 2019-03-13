import tensorflow as tf
import numpy as np

state = tf.Variable(0, name='counter')
# print(state.name)
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.global_variables_initializer() #定义变量必须初始化

with tf.Session() as sess:
    sess.run(init)
    for i in range(3):
        sess.run(update)
        print(sess.run(state))

