import _pickle as cPickle
import numpy
import math
import matplotlib.pyplot as plt
import random


def calculate_angle(valence, arousal):
    return (math.atan2(arousal, valence) * (180 / math.pi) + 360) % 360


def get_emotion_from_angle(angle):
    return int(angle) // 30


def append_angle_and_emotion(valence_and_arousal):
    angles_list = numpy.zeros((40, 1))
    emotions_list = numpy.zeros((40, 1))
    i = 0
    for line in valence_and_arousal:
        angles_list[i, 0] = "%.2f" % calculate_angle(line[0], line[1])
        emotions_list[i, 0] = get_emotion_from_angle(angles_list[i, 0])
        i += 1
    return numpy.append(numpy.append(valence_and_arousal, angles_list, 1), emotions_list, 1)


def save_all_labels_into_file():
    x = cPickle.load(open('D:\eng\data_preprocessed_python\s01.dat', 'rb'), encoding='latin1')  # dict
    labels = x['labels']
    valence_and_arousal = (labels[:, [0, 1]] - 5) / 4
    labelswithanglesandemotions = append_angle_and_emotion(valence_and_arousal)
    for i in range(2, 33):
        x = cPickle.load(
            open('D:\eng\data_preprocessed_python\s' + str(i).zfill(2) + '.dat', 'rb'), encoding='latin1')  # dict
        labels = x['labels']  # numpy.ndarray
        # valence,arousal,dominance,liking
        valence_and_arousal = (labels[:, [0, 1]] - 5) / 4
        labelswithanglesandemotions = numpy.concatenate(
            (labelswithanglesandemotions, append_angle_and_emotion(valence_and_arousal)), 0)
    with open('D:\eng\data_preprocessed_python\labels.txt', 'wb') as outfile:
        numpy.savetxt(outfile, labelswithanglesandemotions, "%.2f")


def prepare_dataset():
    dataset = numpy.empty([1280, 504])
    k = 0
    for i in range(1, 33):
        x = cPickle.load(open('D:\eng\data_preprocessed_python\s' + str(i).zfill(2) + '.dat', 'rb'), encoding='latin1')
        # 37-GSR 39-Plet
        data = x['data']
        gsr = data[:, 36, ::32]
        plet = data[:, 38, ::32]
        for j in range(0, 40):
            dataset[k, :] = numpy.concatenate((gsr[j, :], plet[j, :]), 0)
            k += 1
    with open('D:\eng\data_preprocessed_python\dataset.txt', 'wb') as outfile:
        numpy.savetxt(outfile, dataset, "%.2f")


def prepare_hot_vectors_from_labels(labels):
    vectors = numpy.zeros((1280, 12), dtype=int)
    for i in range(0, 1280):
        vectors[i, int(labels[i, 3])] = 1
    return vectors


def split_dataset_into_training_and_test(dataset, labels):
    test_labels = numpy.zeros((256, 12), dtype=int)
    test_dataset = numpy.zeros((256, 504))
    for i in range(0, 256):
        random_row = random.randint(0, dataset.shape[0] - 1)
        test_dataset[i, :] = dataset[random_row, :]
        test_labels[i, :] = labels[random_row, :]
        dataset = numpy.delete(dataset, random_row, 0)
        labels = numpy.delete(labels, random_row, 0)
    return dataset, labels, test_dataset, test_labels
# emotionsList = ['Pleased','Happy','Excited','Annoyed','Angry','Nervous','Sad','Bored','Sleepy','Calm','Peaceful','Relaxed']


numpy.set_printoptions(suppress=True)
labels = numpy.loadtxt(open('D:\eng\data_preprocessed_python\hot_vectors.txt', 'rb'))
dataset = numpy.loadtxt(open('D:\eng\data_preprocessed_python\dataset.txt', 'rb'))



