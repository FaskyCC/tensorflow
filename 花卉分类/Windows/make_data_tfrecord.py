#encoding=utf-8
import os
import tensorflow as tf
from PIL import Image
import glob

# 原始图片训练集的存储位置
# orig_picture = "F:/flowertf/flower/test"
# 验证集图片存储位置
# val_orig_picture = "F:/flowertf/flower/test"
# 生成图片的存储位置
gen_picture = "C:/Users/Fa/Desktop/Test3/flower_sample/daisy"

# train_record=os.getcwd()+r"\train\flower.tfrecord"
# val_record=os.getcwd()+r"\train\validation_flower.tfrecord"
test_record="C:/Users/Fa/Desktop/Test3/flower_sample/qf/test_flower.tfrecord"

#制作二进制数据
size=299
#创建训练集的Tfrecord
def create_train_record(fname,size):
    classes = {'daisy':0, 'dandelion':1, 'rose':2, 'sunflower':3, 'tulip':4}
    writer = tf.python_io.TFRecordWriter(fname)
    #读取花名和对应的索引编号
    for name,index in classes.items():
        class_path = orig_picture +"/"+ name+"/"
        #获取文件夹下每个文件的名称
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((size, size))
            img_raw = img.tobytes() #将图片转化为原生bytes
            #print index,img_raw
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }
                )
            )
            writer.write(example.SerializeToString())
    writer.close()

def create_val_record(fname,size):
    classes = {'daisy':0, 'dandelion':1, 'rose':2, 'sunflower':3, 'tulip':4}
    writer = tf.python_io.TFRecordWriter(fname)
    #读取花名和对应的索引编号
    for name,index in classes.items():
        class_path = val_orig_picture +"/"+ name+"/"
        #获取文件夹下每个文件的名称
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((size, size))
            img_raw = img.tobytes() #将图片转化为原生bytes
            #print index,img_raw
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }
                )
            )
            writer.write(example.SerializeToString())
    writer.close()
# 创建测试集的Tfrecord
def create_test_record(fname,size):
    count=0
    writer = tf.python_io.TFRecordWriter(fname)
    #获取测试图片数量
    for im in glob.glob(gen_picture + '/*.jpg'):
        count += 1
    for i in range(count):
        img_path = gen_picture +'/'+str(i)+".jpg"
        img = Image.open(img_path)
        img = img.resize((size, size))
        img_raw = img.tobytes()  # 将图片转化为原生bytes
        # print index,img_raw
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    # "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }
            )
        )
        writer.write(example.SerializeToString())
    writer.close()
# create_test_record()
def read_and_decode(filename,is_batch,batch_size=64):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        })
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img,[size,size,3])
    img = tf.cast(img, tf.float32)
    label = tf.cast(label, tf.int64)

    if is_batch:
        min_after_dequeue = 100
        # capacity = min_after_dequeue + 3 * batch_size
        img, label = tf.train.shuffle_batch([img, label],
                                            batch_size=batch_size,
                                            num_threads=3,
                                            capacity=4000,
                                            min_after_dequeue=3500)
    # print(img)
    # print(label)
    #返回的是张量类型的数组
    return img, label

def read_test_and_decode(filename, is_batch,batch_size):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            # 'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        })
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [size, size, 3])
    # img = tf.cast(img, tf.float32)
    img = tf.cast(img, tf.float32)

    if is_batch:
        # min_after_dequeue = 10
        # capacity = min_after_dequeue + 3 * batch_size
        img = tf.train.batch([img], batch_size=batch_size,
                                            capacity=424,
                                            )
    return img

def train_length():
    train_length=0
    classes = {'daisy', 'dandelion', 'rose', 'sunflower', 'tulip'}
    for index, name in enumerate(classes):
        class_path = orig_picture + "/" + name + "/"
        for img_name in os.listdir(class_path):
            train_length+=1
    return train_length

def validation_length():
    validation_length = 0
    classes = {'daisy', 'dandelion', 'rose', 'sunflower', 'tulip'}
    for index, name in enumerate(classes):
        class_path = val_orig_picture + "/" + name + "/"
        for img_name in os.listdir(class_path):
            validation_length += 1
    return validation_length

def test_length():
    test_length = 0
    for im in glob.glob(gen_picture + '/*.jpg'):
        test_length += 1
    return test_length

def main():
    # create_train_record(train_record, 299)
    # create_val_record(val_record, 299)
    create_test_record(test_record,299)

if __name__ == '__main__':
    main()