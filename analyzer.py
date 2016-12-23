import tensorflow as tf
import pylab
import numpy
import sys
import wave
import os
from scipy.stats import mode
from collections import deque
from PIL import Image, ImageChops
from ConfigParser import ConfigParser

"""
This script will generate simple stats on an audio file, given a trained TensorFlow model

Usage: python classifier.py 
"""

def read_and_predict(model_path, wav_path, frame_length, amp_threshold, logging_freq):
    # set up variables for model to be read into
    tensor_size, num_classes, classnames = get_config_data(model_path)
    x = tf.placeholder(tf.float32, [None, tensor_size])
    W = tf.Variable(tf.zeros([tensor_size, num_classes]), name="weights")
    b = tf.Variable(tf.zeros([num_classes]), name="bias")
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    prediction = tf.argmax(y,1)
    sess = tf.Session()

    # restore trained & saved tensorflow model
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    # information about the wav file
    wav = wave.open(wav_path, 'r')
    num_frames = wav.getnframes()
    sample_rate = wav.getframerate()
    num_windows = num_frames/frame_length

    # variables for analysis
    deque_size = 20
    predictions = deque([], deque_size)
    points_per_class = dict((classname, 0) for classname in classnames)
    start_points = 0
    talking_started = False

    print 'Starting classification... Examining ' + str(num_windows) + ' windows.'
    while wav.tell() + frame_length <= num_frames:
        # logging stats
        if (wav.tell() > 0 and wav.tell() % (sample_rate * logging_freq) == 0):
            output_stats(wav.tell(), sample_rate, points_per_class)

        # dump points from start of talking into most predicted class so far,
        # only once predictions deque reaches half capacity
        if len(predictions) == deque_size/2 and talking_started:
            m = mode(predictions)
            points_per_class[classnames[m.mode[0]]] += start_points
            # set talking_started back to false to avoid multiple-counting here
            talking_started = False

        # read frame data from wav file
        frames = wav.readframes(frame_length)
        sound_info = pylab.fromstring(frames, 'Int16')
        amps = numpy.absolute(sound_info)

        # if talking has begun, award silent frames to current speaker
        if amps.mean() < amp_threshold and len(predictions) >= deque_size/2:
            m = mode(predictions)
            points_per_class[classnames[m.mode[0]]] += 1
            continue
        elif amps.mean() < amp_threshold and talking_started:
            start_points += 1
            continue
        elif amps.mean() < amp_threshold:
            continue

        # frame is valid, make prediction
        flat_spectro = create_flat_spectrogram(sound_info, frame_length, sample_rate)

        predicted = prediction.eval(session=sess, feed_dict={x: flat_spectro})
        predictions.append(predicted[0])
        if len(predictions) >= deque_size/2:
            m = mode(predictions)
            points_per_class[classnames[m.mode[0]]] += 1
        else:
            talking_started = True
            start_points += 1

    if os.path.isfile('tmp.png'): os.remove('tmp.png')
    # account for delay by awarding final frames according to mode of predictions list
    points_per_class = finish_analysis(deque_size, predictions, points_per_class, classnames)
    output_stats(num_frames, sample_rate, points_per_class)
    return 


def get_config_data(model_path):
    config = ConfigParser()
    config.read(model_path + '-config.ini')
    tensor_size = int(config.get('Sizes', 'tensor_size'))
    num_classes = int(config.get('Sizes', 'num_classes'))
    classnames_str = config.get('Classnames', 'classnames')
    classnames = classnames_str.split(',')
    return tensor_size, num_classes, classnames


def create_flat_spectrogram(sound_info, frame_length, sample_rate):
    pylab.figure(num=None, figsize=(19, 12))
    pylab.axis('off')
    pylab.specgram(sound_info, NFFT=frame_length, Fs=sample_rate)
    pylab.savefig('tmp.png')
    pylab.close()
    spectro = Image.open('tmp.png')
    spectro = squarify(spectro, 256)
    flat_spectro = flatten(spectro)
    return flat_spectro

def flatten(im):
    data = numpy.array(im)
    flat = data.flatten()
    # tensor placeholder will expect floats
    flat = flat.astype(numpy.float32)
    flat = numpy.multiply(flat, 1.0 / 255.0)
    flat_in_array = []
    flat_in_array.append(flat)
    return flat_in_array


def squarify(im, image_size):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        im = im.crop(bbox)
    im = im.resize((image_size, image_size))
    return im


def output_stats(num_frames, sample_rate, points_per_class):
    total_points = sum(points_per_class.values())
    total_seconds = num_frames / sample_rate

    print points_per_class
    stats_string = '======================STATS=======================\n'
    stats_string += 'Total time ' + get_readable_time(total_seconds) + '\n'
    for key, value in points_per_class.iteritems():
        stats_string += '--------------------------------------------------\n'
        class_proportion = 0 if total_points == 0 else float(value) / float(total_points)
        class_total_seconds = class_proportion * total_seconds
        stats_string += key + ' spoke for ' + get_readable_time(int(class_total_seconds)) + '\n'
    stats_string += '=================================================='
    print stats_string


def get_readable_time(total_seconds):
    seconds = '0' + str(total_seconds % 60) if total_seconds % 60 < 10 else str(total_seconds % 60)
    total_minutes = total_seconds / 60
    minutes = '0' + str(total_minutes % 60) if total_minutes % 60 < 10 else str(total_minutes % 60)
    total_hours = total_minutes / 60
    hours = '0' + str(total_hours % 24) if total_hours % 24 < 10 else str(total_hours % 24)
    return str(hours) + ':' + str(minutes) + ':' + str(seconds)


def finish_analysis(deque_size, predictions, points_per_class, classnames):
    while len(predictions) > deque_size / 2:
        predictions.popleft()
        m = mode(predictions)
        points_per_class[classnames[m.mode[0]]] += 1
    return points_per_class


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print "Incorrect usage, please see top of file."
        exit()
    model_path = sys.argv[1]
    wav_path = sys.argv[2]
    frame_length = int(sys.argv[3])
    amp_threshold = int(sys.argv[4])
    if (len(sys.argv) == 6):
        logging_freq = int(sys.argv[5])
    else:
        # log every minute by default
        logging_freq = 60
    read_and_predict(model_path, wav_path, frame_length, amp_threshold, logging_freq)
