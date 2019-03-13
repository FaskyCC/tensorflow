import tensorflow as tf
import numpy as np

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])
product = tf.matmul(matrix1,matrix2)  #矩阵乘法,np.dot(m1,m2)

# #方法1
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()

# 方法2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)