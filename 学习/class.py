import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

#Mnist
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

# 定义一个用来添加网络层的函数,参数包括输入数据,输入数据大小,输出数据大小,可选择的激活函数
def add_layer(inputs, in_size, out_size, activation_function = None):
    '''
    :param inputs:
    :param in_size:
    :param out_size:
    :param activation_function:
    :return:
    '''
    # y = wx+b

    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs, Weights)+biases

    #对输出数据运用激活函数
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


epoch =10
batch_size =100
total_batch = int(mnist.train.num_examples/batch_size)


#定义placeholder
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

#定义输出层
'''
两层 batch = 100  0.001
两层 batch = 1 

'''

l1 = add_layer(xs, 784, 500, activation_function=None)
prediction = add_layer(l1, 500, 10, activation_function=None)

#定义loss

cross_entropy = -tf.reduce_sum(ys*tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
tf.summary.scalar('loss', cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={
            xs: batch_xs, ys: batch_ys
        })
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={
            xs: batch_xs, ys: batch_ys
        })
        writer.add_summary(result, i)
        print(i, compute_accuracy(mnist.test.images, mnist.test.labels))
        #   print(sess.run(cross_entropy,feed_dict={xs: batch_xs, ys: batch_ys}))
