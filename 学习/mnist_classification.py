import tensorflow as tf
# 导入MNIST数字手写体数据库
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)

# 定义输入层、隐含层、输出层的神经元个数
in_units = 784
h1_units = 300
out_units = 10

# 定义输入层，keep_prob是dropout的比例
with tf.name_scope("input"):
    x = tf.placeholder(tf.float32,[None,in_units])
    y_= tf.placeholder(tf.float32,[None,out_units])
keep_prob = tf.placeholder(tf.float32)

# 定义隐含层的权重、偏置、激活函数
with tf.name_scope("hidden_layer1"):
    with tf.name_scope("w1"):
        w1 = tf.Variable(tf.random_normal([in_units,h1_units],stddev = 0.1))
        tf.summary.histogram('Weight1',w1)
    with tf.name_scope("b1"):
        b1 = tf.Variable(tf.zeros([h1_units])) + 0.01
        tf.summary.histogram('biases1',b1)
    with tf.name_scope("w1_b1"):
        hidden1 = tf.nn.relu(tf.matmul(x,w1) + b1)
        tf.summary.histogram('output1',hidden1)


# 定义输出层的权重、偏置、激活函数
with tf.name_scope("output_layer"):
    with tf.name_scope("w2"):
        w2 = tf.Variable(tf.random_normal([h1_units,out_units],stddev = 0.1))
        tf.summary.histogram('Weight2',w2)
    with tf.name_scope("b2"):
        b2 = tf.Variable(tf.zeros([out_units]))
        tf.summary.histogram('biases2',b2)
    with tf.name_scope("w2_b2"):
        hidden1_drop = tf.nn.dropout(hidden1,keep_prob)
with tf.name_scope("output"):
    y = tf.nn.softmax(tf.matmul(hidden1_drop,w2)+b2)
    tf.summary.histogram('output',y)

# 定义损失函数———交叉熵
with tf.name_scope("loss"):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices = [1]))
    tf.summary.scalar('cross_entropy', cross_entropy)
# 计算准确率
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# 定义优化器——Adagrad，和学习率:0.3
with tf.name_scope("train"):
    train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# 初始化所有的变量
init = tf.global_variables_initializer()
# 开始导入数据，正式计算，迭代3000步，训练时batch size=100
with tf.Session() as sess:
    sess.run(init)
    merge = tf.summary.merge_all()
    writer = tf.summary.FileWriter("log",sess.graph)
    for i in range(3000+1):
        batch_xs,batch_ys = mnist.train.next_batch(1000)
        sess.run(train_step,feed_dict = {x:batch_xs,y_:batch_ys,keep_prob:0.75})
        loss_run = sess.run(cross_entropy,feed_dict = {x:batch_xs,y_:batch_ys,keep_prob:0.75})
        accuracy_run = sess.run(accuracy,feed_dict = {x:batch_xs,y_:batch_ys,keep_prob:0.75})
        print('after %d steps training steps,the loss is %g and the accuracy is %g'%(i,loss_run,accuracy_run))
        result = sess.run(merge,feed_dict = {x:batch_xs,y_:batch_ys,keep_prob:1})
        writer.add_summary(result,i)
        # 训练完后直接加载测试集数据，进行测试
        if i %100 == 0:
            loss_run = sess.run(cross_entropy,feed_dict = {x:mnist.test.images,y_:mnist.test.labels,keep_prob:1})
            accuracy_run = sess.run(accuracy,feed_dict = {x:mnist.test.images,y_:mnist.test.labels,keep_prob:1})
            print('the loss in test dataset is %g and the accuracy in test dataset is %g'%(loss_run,accuracy_run))
