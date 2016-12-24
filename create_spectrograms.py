"""
This script creates a bunch of square spectrograms from short audio clips.

Usage: python create_spectrograms.py [class dir] [destination] [frame length]
                                     [image size] [amp threshold]
where
    [class dir] is the name of the directory containing directories named for
        each class, which in turn contain wav files to be turned into spectrograms
        (this should include the final slash, e.g. path/to/directory/)
    [destination] is the directory to save the spectrograms to (this should 
        include the final slash, e.g. path/to/directory/)
    [frame length] is how long each frame to be FFT'd should be
    [image size] is the height and width of resulting spectrogram images
    [amp threshold] is the minimum average amplitude (sample value) of each frame that
        will be treated as a valid frame and made a spectrogram of
"""

import sys
import os
import wave
from glob import glob
from helper import *


def create_spectrograms(class_dir, destination, frame_length, image_size, amp_threshold):
    classnames = [os.path.basename(clas) for clas in glob(class_dir + '*')]
    total_count = 0
    for classname in classnames:
        print '=================================================='
        print 'Creating spectrograms for class "' + classname + '"'
        print '=================================================='
        create_class_dirs(destination, classname)
        wavs = glob(class_dir + classname + '/*.wav')
        class_count = 0
        for wav_file in wavs:
            wav = wave.open(wav_file, 'r')
            num_frames = wav.getnframes()
            sample_rate = wav.getframerate()
            num_windows = num_frames/frame_length
            print 'Now creating spectrograms for file ' + wav_file + '... Examining ' + str(num_windows) + ' windows.'
            while (wav.tell() + frame_length) < num_frames:
                frames = wav.readframes(frame_length)
                sound_info = pylab.fromstring(frames, 'Int16')
                amps = numpy.absolute(sound_info)

                # ignore windows with low amplitudes (i.e. lack of vocal information)
                if (amps.mean() < amp_threshold):
                    continue

                # split training:testing 7:1
                if (class_count > 1):
                    filename = str(class_count/7) if (class_count % 7 == 0) else str(class_count - (class_count/7) - 1)
                else:
                    filename = '0'
                full_dest = destination + 'testing/' + classname + '/' if (class_count % 7 == 0) else destination + 'training/' + classname + '/'

                save_new_spectrogram(sound_info, frame_length, sample_rate, filename, image_size, full_dest)

                class_count += 1
                total_count += 1
            wav.close()
        print 'Finished creating spectrograms for class "' + classname + '". ' + str(class_count) + ' spectrograms created for this class.'
    print '--------------------------------------------------'
    print 'All finished! A total of ' + str(total_count) + ' spectrograms was created.' 


def create_class_dirs(destination, classname):
    # create training/testing directories if non-existent
    if not os.path.isdir(destination + 'training'):
        os.mkdir(destination + 'training')
    if not os.path.isdir(destination + 'testing'):
        os.mkdir(destination + 'testing')
    # put class directories in training/testing dirs if non-existent
    if not os.path.isdir(destination + 'training/' + classname):
        os.mkdir(destination + 'training/' + classname)
    if not os.path.isdir(destination + 'testing/' + classname):
        os.mkdir(destination + 'testing/' + classname)


def save_new_spectrogram(sound_info, frame_length, sample_rate, filename, image_size, full_dest):
    spectro = create_spectrogram(sound_info, frame_length, sample_rate, filename, image_size)
    spectro.save(filename + '.png')
    os.rename(filename + '.png', full_dest + filename + '.png')
    return


if __name__ == "__main__":
    class_dir = sys.argv[1]
    destination = sys.argv[2]
    frame_length = sys.argv[3]
    image_size = sys.argv[4]
    amp_threshold = sys.argv[5]
    create_spectrograms(class_dir, destination, int(frame_length), int(image_size), int(amp_threshold))
