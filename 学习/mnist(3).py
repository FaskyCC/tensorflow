import tensorflow as tf

# 导入MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)

# 定义输入层、隐含层、输出层的神经元个数
input = 784
hidden1 = 300
hidden2 = 100
output = 10
epoch_size = 50
batch_size = 1000
batch_num = int(mnist.train.num_examples/batch_size)
dropout = 1

def add_layer(inputs, in_size, out_size, layer_name, activation_function = None):
    # 定义隐含层的权重、偏置、激活函数
    with tf.name_scope(layer_name):
        with tf.name_scope("weight"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size], stddev=0.1))
            tf.summary.histogram('Weight', Weights)
        with tf.name_scope("biase"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.01)
            tf.summary.histogram('biases', biases)
        with tf.name_scope("Wx_b"):
            output = tf.matmul(inputs, Weights) + biases
            if activation_function is None:
                return output
            else:
                output = activation_function(output)
                return output


# 定义输入层，keep_prob是dropout的比例
with tf.name_scope("input"):
    xs = tf.placeholder(tf.float32, [None, input])
    ys = tf.placeholder(tf.float32, [None, output])
keep_prob = tf.placeholder(tf.float32)

# 定义隐含层
# layer1 = add_layer(xs, input, hidden1, 'layer1', activation_function=tf.nn.relu)
# layer2 = add_layer(layer1, hidden1, hidden2,'layer2',activation_function=tf.nn.relu)
#定义输出层
prediction = add_layer(xs, input, output, 'layer3', activation_function=tf.nn.softmax)

# 定义损失函数———交叉熵
with tf.name_scope("loss"):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
    tf.summary.scalar('cross_entropy', cross_entropy)
# 计算准确率
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 定义优化器和学习率
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)


# 初始化所有的变量
init = tf.global_variables_initializer()
# 开始导入数据，正式计算，迭代3000步，训练时batch size=100
with tf.Session() as sess:
    sess.run(init)
    merge = tf.summary.merge_all()
    writer = tf.summary.FileWriter("log", sess.graph)
    for i in range(epoch_size*batch_num+1):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: dropout})

        result = sess.run(merge, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1})
        writer.add_summary(result, i)
        # 训练完加载测试集数据，进行测试
        if i % 100 == 0:
            loss_run = sess.run(cross_entropy, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: dropout})
            accuracy_run = sess.run(accuracy, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: dropout})
            print('After %d steps training steps,The loss is %g and The accuracy is %g' % (i, loss_run, accuracy_run))
            loss_run = sess.run(cross_entropy, feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1})
            accuracy_run = sess.run(accuracy,feed_dict={xs: mnist.test.images,ys: mnist.test.labels, keep_prob: 1})
            print('The loss in test dataset is %g and The accuracy in test dataset is %g' % (loss_run, accuracy_run))
    accuracy_run = sess.run(accuracy, feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1})
    print('The final accuracy in test dataset is %g' % (accuracy_run))
