import tensorflow as tf
import os
import sys
import random
from glob import glob
from helper import *

"""
This script contains methods for training and testing a tensorflow model
using spectrogram data as input.

Usage: python train.py [filepath] [train loops] [train samples] [test loops] [test samples] [?savepath]
where 
[filepath] is the location of the data
[train loops] is the number of times to run the training loop
[samples per loop] is the number of samples of each class to be fed to
    the model every iteration of the training loop
[test loops] is the number of times to run the test loop
[test samples] is the number of samples of each class to be fed to
    the model every iteration of the testing loop
[savepath] is the path of the file to save the variables/tensors to,
    if you want to save the model after trianing (this parameter is optional)
"""


def train_test_save(datapath, train_loops, train_samples, test_loops, test_samples, save_path = None):
    training_dir = datapath + "training/"
    tensor_size, classnames, num_classes = get_data_info(training_dir)

    # set up our model
    x = tf.placeholder(tf.float32, [None, tensor_size])
    W = tf.Variable(tf.zeros([tensor_size, num_classes]), name="weights")
    b = tf.Variable(tf.zeros([num_classes]), name="bias")
    y = tf.matmul(x, W) + b
    y_ = tf.placeholder(tf.float32, [None, num_classes])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)

    # for applying dropout on training operation
    keep_prob = tf.placeholder(tf.float32)

    # in each training step, ...
    for step in range(train_loops):
        batch_samples = []
        batch_labels = []
        # each class is set up with its label and correct number of training samples, and...
        for class_no in range(num_classes):
            class_label = [0] * num_classes
            class_label[class_no] = 1
            classname = classnames[class_no]
            num_samples = len(glob(training_dir + classname + '/*.png'))
            indices = numpy.arange(num_samples)
            numpy.random.shuffle(indices)
            # a random sample is selected and added to the array of samples in the batch
            for sample in range(train_samples):
                im = training_dir + classname + '/' + str(indices[sample]) + '.png'
                spectro = Image.open(im)
                flat = flatten_image(spectro)
                batch_samples.append(flat)
                batch_labels.append(class_label)
        train_accuracy = sess.run(accuracy, feed_dict={x:batch_samples, y_: batch_labels, keep_prob: 1.0})
        print 'Step: ' + str(step+1) + ', training accuracy: ' + str(train_accuracy)
        sess.run(train_step, feed_dict={x: batch_samples, y_: batch_labels, keep_prob: 0.5})

    # test accuracy of the model once trained
    accuracies = []
    for test in range(test_loops):
        testing_data, testing_labels = get_test_data(datapath, classnames, test_samples)
        success_rate = sess.run(accuracy, feed_dict={x: testing_data, y_: testing_labels, keep_prob: 1.0})
        print 'Test number: ' + str(test+1) + '. Success rate: ' + str(success_rate*100) + '%'
        accuracies.append(success_rate)
    avg = numpy.mean(accuracies)
    print 'This model tested with an average success rate of ' + str(avg*100) + '%'

    # save model to path specified (if specified at all)
    if (save_path != None):
        make_config_file(save_path, tensor_size, num_classes, classnames)
        saver.save(sess, save_path)


def get_data_info(datapath):
    classnames = [os.path.basename(clas) for clas in glob(datapath + '*')]
    first_classname = classnames[0]
    spectro_width, spectro_height = Image.open(datapath + first_classname + '/' + '0.png').size

    # verify that spectrograms are (probably) all the same size
    for classname in classnames:
        num_items = len(glob(datapath + classname + '/*.png'))
        rand_item = random.randrange(0, num_items)
        width, height = Image.open(datapath + classname + '/' + str(rand_item) + ".png").size
        assert (width == spectro_width and height == spectro_height), "Image size error: Found two spectrograms with differing dimensions"

    # 4 values per pixel (red, green, blue, opacity)
    tensor_size = 4 * spectro_width * spectro_height
    num_classes = len(classnames)
    return tensor_size, classnames, num_classes


def get_test_data(datapath, classnames, test_samples):
    testing_dir = datapath + 'testing/'
    testing_data = []
    testing_labels = []
    # each class...
    for class_no in range(len(classnames)):
        class_label = [0] * len(classnames)
        class_label[class_no] = 1
        # randomize test samples chosen
        indices = numpy.arange(len(glob(testing_dir + classnames[class_no] + '/*.png')))
        numpy.random.shuffle(indices)
        class_dir = testing_dir + classnames[class_no] + '/'
        total_test_samples = len(glob(class_dir + '*.png'))
        samples_to_get = test_samples if (test_samples < total_test_samples) else total_test_samples
        # each sample in the class
        for sample in range(samples_to_get):
            im = class_dir + str(indices[sample]) + '.png'
            spectro = Image.open(im)
            flat = flatten_image(spectro)
            testing_data.append(flat)
            testing_labels.append(class_label)
    return testing_data, testing_labels


def make_config_file(path, tensor_size, num_classes, classnames):
    config = ConfigParser()
    cfg_file = open(path + '-config.ini', 'w+')
    config.add_section('Sizes')
    config.set('Sizes', 'tensor_size', tensor_size)
    config.set('Sizes', 'num_classes', num_classes)
    config.add_section('Classnames')
    class_string = ''
    for classname in classnames:
        class_string = class_string + classname
        if classname == classnames[-1]:
            config.set('Classnames', 'classnames', class_string)
            break
        class_string = class_string + ','
    config.write(cfg_file)
    cfg_file.close()


if __name__ == "__main__":
    # make sure correct number of arguments
    if len(sys.argv) < 6:
        print "Incorrect usage"
        exit()
    datapath = sys.argv[1]
    train_loops = sys.argv[2]
    train_samples = sys.argv[3]
    test_loops = sys.argv[4]
    test_samples = sys.argv[5]
    if (len(sys.argv) == 7):
        save_path = sys.argv[6]
    else: save_path = None
    train_test_save(datapath, int(train_loops), int(train_samples), int(test_loops), int(test_samples), save_path)

