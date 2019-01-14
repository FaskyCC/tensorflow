import tensorflow as tf
import inception_preprocessing
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import time
import os
from train_flowers import get_split
import matplotlib.pyplot as plt
from make_data_tfrecord import read_test_and_decode
from train_flowers import get_split, load_batch
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
import inception_preprocessing
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import time
import os
from train_flowers import get_split, load_batch
import matplotlib.pyplot as plt

plt.style.use('ggplot')
slim = tf.contrib.slim
image_size = 299
# State your log directory where you can retrieve your model
log_dir = r'C:/Users/Fa/Desktop/Test/'

# Create a new evaluation log directory to visualize the validation process
log_eval =  'C:/Users/Fa/Desktop/Test/log_eval_test'

# State the dataset directory where the validation set is found
dataset_dir =  r'C:/Users/Fa/Desktop/Test/'

# State the batch_size to evaluate each time, which can be a lot more than the training batch
batch_size = 53

# State the number of epochs to evaluate
num_epochs = 3

# Get the latest checkpoint file
checkpoint_file = tf.train.latest_checkpoint(log_dir)

# def load_batch_2(dataset, batch_size, height=image_size, width=image_size, is_training =True):
#     '''
#     Loads a batch for training.
#     INPUTS:
#     - dataset(Dataset): a Dataset class object that is created from the get_split function
#     - batch_size(int): determines how big of a batch to train
#     - height(int): the height of the image to resize to during preprocessing
#     - width(int): the width of the image to resize to during preprocessing
#     - is_training(bool): to determine whether to perform a training or evaluation preprocessing
#     OUTPUTS:
#     - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
#     - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).
#     '''
#     # First create the data_provider object
#     data_provider = slim.dataset_data_provider.DatasetDataProvider(
#         dataset,
#         common_queue_capacity=24 + 3 * batch_size,
#         common_queue_min=24)
#
#     # Obtain the raw image using the get method
#     raw_image = data_provider.get(['image'])
#
#     # Perform the correct preprocessing for this image depending if it is training or evaluating
#     image = inception_preprocessing.preprocess_image(raw_image, height, width, is_training)
#
#     # As for the raw images, we just do a simple reshape to batch it up
#     raw_image = tf.expand_dims(raw_image, 0)
#     raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
#     raw_image = tf.squeeze(raw_image)
#
#     # Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
#     images, raw_images = tf.train.batch(
#         [image, raw_image],
#         batch_size=batch_size,
#         num_threads=4,
#         capacity=4 * batch_size,
#         allow_smaller_final_batch=True)
#
#     return images, raw_images



def run():
    # Create log_dir for evaluation information
    if not os.path.exists(log_eval):
        os.mkdir(log_eval)

    # Just construct the graph from scratch again
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)
        # Get the dataset first and load one batch of validation images and labels tensors. Set is_training as False so as to use the evaluation preprocessing

        images = read_test_and_decode("C:/Users/Fa/Desktop/Test3/test_flower.tfrecord",False,299)
        images=tf.train.batch([images],batch_size=batch_size,capacity=424)
        # images, raw_images = load_batch_2(dataset, batch_size=batch_size, is_training=True)

        # Create some information about the training steps

        num_steps_per_epoch = 8

        # Now create the inference model but set is_training=False
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(images, num_classes=5, is_training=True)

        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        # Just define the metrics to track without the loss or whatsoever
        predictions = tf.argmax(end_points['Predictions'], 1)
        sv = tf.train.Supervisor(logdir=log_eval, summary_op=None, saver=None, init_fn=restore_fn)
        with sv.managed_session() as sess:
            # sess.run(tf.global_variables_initializer())
            for i in range(53):
                predict_class=sess.run(predictions)
                for i in range(batch_size):
                    predicPer = predict_class[i]
                    prediction_name = dataset.labels_to_name[predicPer]
                    print(prediction_name)




if __name__ == '__main__':
    run()