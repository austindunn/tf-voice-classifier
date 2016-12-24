import tensorflow as tf
import os
import sys
from glob import glob
from helper import *

"""
This script is used to load previously trained TF models and allow them to
    evaluate real data.

Usage: python test_model.py [model path] [test path] [num test samples]
where
    [model path] is the path to a trained and saved TensorFlow model
    [test path] is the path to the directory containing test data, organized
        in subdirectories named for each class
    [num test samples] is the number of samples (spectrograms) to look at
        for each class
"""

def test(model_path, test_path, num_test_samples):
    tensor_size, num_classes, classnames = get_config_data(model_path)
    x = tf.placeholder(tf.float32, [None, tensor_size])
    W = tf.Variable(tf.zeros([tensor_size, num_classes]), name="weights")
    b = tf.Variable(tf.zeros([num_classes]), name="bias")
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    prediction = tf.argmax(y,1)
    sess = tf.Session()

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    # initialize confusion matrix (as dictionary)
    # first name is correct answer, second name is the prediction
    confusion_dict = {}
    for correct_class in classnames:
        for guess_class in classnames:
            confusion_dict[correct_class + '_' + guess_class] = 0

    for classname in classnames:
        print '--------------------------------------------------'
        print 'Looking at files for class: "' + classname + '"'
        indices = numpy.arange(len(glob(test_path + classname + '/*.png')))
        numpy.random.shuffle(indices)
        # numpy.random.shuffle(indices)
        class_dir = test_path + classname + '/'
        total_test_samples = len(glob(class_dir + '*.png'))
        samples_to_get = num_test_samples if (num_test_samples < total_test_samples) else total_test_samples
        # each sample in the class
        for sample in range(samples_to_get):
            im = class_dir + str(indices[sample]) + '.png'
            spectro = Image.open(im)
            flat = flatten_image(spectro)
            predicted = prediction.eval(session=sess, feed_dict={x: [flat]})
            confusion_key = classname + '_' + classnames[predicted[0]]
            confusion_dict[confusion_key] += 1
        print 'Correct guesses for class "' + classname + ': ' + str(confusion_dict[classname + '_' + classname]) + '/' + str(samples_to_get)

    print '=================================================='
    print 'Here\'s the confusion dictionary: first name in the key is the correct answer, second is the prediction made by the model.'
    for key, value in sorted(confusion_dict.items()):
        print key + ': ' + str(value)
    return


if __name__ == "__main__":
    assert len(sys.argv) == 4, 'Incorrect usage, please see top of test_model.py file.'

    model_path = sys.argv[1]
    test_path = sys.argv[2]
    num_test_samples = int(sys.argv[3])
    test(model_path, test_path, num_test_samples)
