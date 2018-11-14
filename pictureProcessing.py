import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# 数据集的路径
INPUT_DATA = r'/Users/wangqingfa/Desktop/kaggle/train'
# 分割好的数据集
OUT_FILE = r'/Users/wangqingfa/Desktop/kaggle/flower_processed_data.npy'

# 测试数据和验证数据所占的比例为10%
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10


def create_image_lists(sess, testing_percentage, validation_percentage):
    # 读取数据集文件夹内的几个文件夹
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True

    # 初始化各个数据集
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0
    current_image = 0

    # 读取所有的子目录
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extension = 'jpg'
        file_list = []
        # 获取图片所属的类别文件夹
        dir_name = os.path.basename(sub_dir)

        # 读取文件夹下*.jpg的文件名
        file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
        # 读取名字为上面类型的文件的名字，保存到列表中
        file_list.extend(glob.glob(file_glob))

        for file_name in file_list:
            current_image = current_image + 1
            print(current_image)
            # 利用tensorflow的方法以二进制的格式读取图像
            image_raw_data = gfile.FastGFile(file_name, 'rb').read()
            # 对上面的二进制图像进行解码
            image = tf.image.decode_jpeg(image_raw_data)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            # resize图片大小
            image = tf.image.resize_images(image, [229, 229])
            image_value = sess.run(image)

            # 随机划分数据集
            # 随机生成一个0-100的数
            chance = np.random.randint(100)
            # 根据比例划分数据集
            if chance < validation_percentage:
                validation_images.append(image_value)
                validation_labels.append(current_label)
            elif chance < (validation_percentage + testing_percentage):
                testing_images.append(image_value)
                testing_labels.append(current_label)
            else:
                training_images.append(image_value)
                training_labels.append(current_label)
        current_label += 1
    # 打乱训练集数据
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)
    return np.asarray([training_images, training_labels,
                       validation_images, validation_labels,
                       testing_images, testing_labels])


# 定义主函数
def main():
    with tf.Session() as sess:
        processed_data = create_image_lists(sess, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
        np.save(OUT_FILE, processed_data)


if __name__ == '__main__':
    main()