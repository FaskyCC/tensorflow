import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')

# 添加一层网格

# 定义一个用来添加网络层的函数,参数包括输入数据,输入数据大小,输出数据大小,可选择的激活函数
def add_layer(inputs, in_size, out_size, n_layer, activation_function = None):
    '''
    :param inputs:
    :param in_size:
    :param out_size:
    :param activation_function:
    :return:
    '''
    # y = wx+b
    layer_name ='layer%s' % n_layer
    with tf. name_scope(layer_name):
        with tf. name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]),name='W')
            tf.summary.histogram(layer_name +'weights',Weights)
        with tf.name_scope('biase'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1)
            tf.summary.histogram(layer_name + 'biase', biases)
        with tf.name_scope('y'):
            Wx_plus_b = tf.matmul(inputs, Weights)+biases

        #对输出数据运用激活函数
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
            tf.summary.histogram(layer_name+'/outputs',outputs)
        return outputs

#构建数据
x_data = np.linspace(-1,1,300)[: ,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')


#(输入层)
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)

#(输出层)
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

#损失函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices = [1]))
    tf.summary.scalar('loss', loss)
#梯度下降优化器
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.1).minimize(loss)
#变量初始化
init = tf.global_variables_initializer()
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/",sess.graph)
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)

plt.ion()


for i in range(1000):
    sess.run(train_step, feed_dict = {xs:x_data, ys:y_data})
    if i % 50 == 0:
        result = sess.run(merged,feed_dict = {
            xs: x_data, ys: y_data
        })
        writer.add_summary(result,i)

        #输出损失
        print(sess.run(loss, feed_dict = {
            xs: x_data, ys: y_data
        }))

        #画拟合曲线
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict = {
            xs:x_data
        })
        lines = ax.plot(x_data, prediction_value,'r-',lw = 5)
        plt.pause(0.1)

plt.ioff()
plt.show()
