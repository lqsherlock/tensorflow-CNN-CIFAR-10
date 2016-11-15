#导入各种包，该程序时用ipython，如果是python需要修改
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os

# Use PrettyTensor to simplify Neural Network construction.
import prettytensor as pt


import sys
sys.path.append('/home/lq/桌面/TensorFlow-Tutorials-master')

#导入cifar10.py文件内容
import cifar10

#导入数据维度等信息
from cifar10 import img_size, num_channels, num_classes
#原始图片的大小时32×32,我们需要剪裁成24×24
img_size_cropped = 24

 
#×××××××××××××××× tensorflow构造阶段  ×××××××××××××××××××××××××

#用于输入图像的占位变量
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

#用于输入图像对应标签的占位变量
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

#y_true是一个大小为10的数组，里面是小于1的小数
#通过argmax函数将其与1对比，结果是最大的数变为1，其他的为0
y_true_cls = tf.argmax(y_true, dimension=1)


#对图像进行预处理
#如果是训练集，则进行随机剪裁，水平翻转，色调/对比度调整
#如果是测试集，则只是在中心周围剪裁
def pre_process_image(image, training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph. 
    if training:
        # For training, add the following to the TensorFlow graph.
        # Randomly crop the input image.
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])
        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)      
        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
        # Some of these functions may overflow and result in pixel
        # values beyond the [0, 1] range. It is unclear from the
        # documentation of TensorFlow 0.10.0rc0 whether this is
        # intended. A simple solution is to limit the range.
        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        # For training, add the following to the TensorFlow graph.
        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                      target_height=img_size_cropped,
                                                       target_width=img_size_cropped)
    return image


#对图像集中的每一个图像进行处理
def pre_process(images, training):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)
    return images

#2层卷积神经网络的构建（利用了Pretty Tensor 框架）
def main_network(images, training):
    # Wrap the input images as a Pretty Tensor object.
    x_pretty = pt.wrap(images)
    # Pretty Tensor uses special numbers to distinguish between
    # the training and testing phases.
    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer
    # Create the convolutional neural network using Pretty Tensor.
    # It is very similar to the previous tutorials, except
    # the use of so-called batch-normalization in the first layer.
    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        y_pred, loss = x_pretty.\
            conv2d(kernel=5, depth=64, name='layer_conv1', batch_normalize=True).\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, depth=64, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=256, name='layer_fc1').\
            fully_connected(size=128, name='layer_fc2').\
            softmax_classifier(class_count=num_classes, labels=y_true)
    return y_pred, loss



#创建并训练神经网络，注意其中有一个变量域（variable-scope）
#为“network”,通过这个变量域我们可以使变量复用
def create_network(training):
    # Wrap the neural network in the scope named 'network'.
    # Create new variables during training, and re-use during testing.
    with tf.variable_scope('network', reuse=not training):
        # Just rename the input placeholder variable for convenience.
        images = x
        # Create TensorFlow graph for pre-processing.
        images = pre_process(images=images, training=training)
        # Create TensorFlow graph for the main processing.
        y_pred, loss = main_network(images=images, training=training)
    return y_pred, loss

#×××××××××××××××××  训练阶段    ×××××××××××××××××××××××
#设置步长
#trainable=False 表示不用优化这个变量
global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)

#创建一个训练阶段的神经网络，并获取损失值
_, loss = create_network(training=True)

#计算损失函数
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)


#×××××××××××××××××  测试阶段    ×××××××××××××××××××××××
#创建一个测试阶段的神经网络，并返回预测结果
y_pred, _ = create_network(training=False)

#计算预测的结果
y_pred_cls = tf.argmax(y_pred, dimension=1)

#计算预测值与实际值是否相等
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

#计算神经网络的精度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#保存神经网络中的变量，下次构建时可以直接读取，而不用再重新训练一次
saver = tf.train.Saver()

 
#×××××××××××××××   恢复tensorflow     ×××××××××××××××××××××××
#创建tensorflow的会话
session = tf.Session()


#如果前边保存了神经网络，下面的代码可以在保存点恢复训练好的神经网络
#设置存储的目录
save_dir = 'data/checkpoints/'
#如果路径不存在就创建对应路径
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = save_dir + 'cifar10_cnn'

try:
    print("Trying to restore last checkpoint ...")
    # Use TensorFlow to find the latest checkpoint - if any.
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
    # Try and load the data in the checkpoint.
    saver.restore(session, save_path=last_chk_path)
    # If we get to this point, the checkpoint was successfully loaded.
    print("Restored checkpoint from:", last_chk_path)
except:
    # If the above failed for some reason, simply
    # initialize all the variables for the TensorFlow graph.
    print("Failed to restore checkpoint. Initializing variables instead.")
 


 
###################    对恢复的神经网络进行测试     #####################################

#如果cifar10已经下载，则直接读取并解压
cifar10.maybe_download_and_extract()

#读取并显示分类的类别
class_names = cifar10.load_class_names()

#读取测试数据集
images_test, cls_test, labels_test = cifar10.load_test_data()

#初始化tensorflow
session.run(tf.initialize_all_variables())


#函数：在3×3的网格中画出9幅图
def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true) == 9
    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)
    # Adjust vertical spacing if we need to print ensemble and best-net.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'
        # Plot image.
        ax.imshow(images[i, :, :, :],
                  interpolation=interpolation)            
        # Name of the true class.
        cls_true_name = class_names[cls_true[i]]
        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # Name of the predicted class.
            cls_pred_name = class_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)      
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])  
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


#该函数可以画出被错分类的图片
def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.
    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.
    # Negate the boolean array.
    incorrect = (correct == False)  
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = images_test[incorrect]   
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]
    # Get the true classes for those images.
    cls_true = cls_test[incorrect]    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


#该函数可以画出混淆矩阵
def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.
    # Print the confusion matrix as text.
    for i in range(num_classes):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)
    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))


# Split the data-set in batches of this size to limit RAM usage.
batch_size = 256

#计算图像预测的类
def predict_cls(images, labels, cls_true):
    # Number of images.
    num_images = len(images)
    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)
    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.
    # The starting index for the next batch is denoted i.
    i = 0
    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)
        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}
        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j
    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)
    return correct, cls_pred


#计算测试集的预测结果
def predict_cls_test():
    return predict_cls(images = images_test,
                       labels = labels_test,
                       cls_true = cls_test)

#计算分类的精度
def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True means 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.    
    # Return the classification accuracy
    # and the number of correct classifications.
    return correct.mean(), correct.sum()

#打印测试集的预测精度
def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls_test()   
    # Classification accuracy and the number of correct classifications.
    acc, num_correct = classification_accuracy(correct)   
    # Number of images being classified.
    num_images = len(correct)
    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))
    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)
    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


#输出识别率，并打印出被错误分类的图
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)



#关闭tensorflow会话
session.close()






