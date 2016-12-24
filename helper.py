import numpy
import pylab
from PIL import Image, ImageChops
from ConfigParser import ConfigParser


def create_spectrogram(sound_info, frame_length, sample_rate, filename, image_size):
    pylab.figure(num=None, figsize=(19, 12))
    pylab.axis('off')
    pylab.specgram(sound_info, NFFT=frame_length, Fs=sample_rate)
    pylab.savefig(filename + '.png')
    pylab.close()

    spectro = Image.open(filename + '.png')
    spectro = squarify(spectro, image_size)
    return spectro


def flatten_image(im):
    data = numpy.array(im)
    flat = data.flatten()
    # tensor placeholder will expect floats
    flat = flat.astype(numpy.float32)
    flat = numpy.multiply(flat, 1.0 / 255.0)
    return flat


def get_config_data(model_path):
    config = ConfigParser()
    config.read(model_path + '-config.ini')
    tensor_size = int(config.get('Sizes', 'tensor_size'))
    num_classes = int(config.get('Sizes', 'num_classes'))
    classnames_str = config.get('Classnames', 'classnames')
    classnames = classnames_str.split(',')
    return tensor_size, num_classes, classnames


def squarify(im, image_size):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        im = im.crop(bbox)
    im = im.resize((image_size, image_size))
    return im
