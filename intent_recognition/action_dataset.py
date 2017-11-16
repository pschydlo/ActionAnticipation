import numpy as np
import os

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

feature_path = ('/media/Data/paul/action_recognition/dataset/joints')

label_path   = ('/media/Data/paul/action_recognition/dataset/actionLabel.txt')


def load():
    features = readFeatures(feature_path)
    labels, dictionary = readLabels(label_path)

    (features, labels, frames) = sync(features, labels)
    #(features, labels) = cutUp(features, labels)

    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size=0.33,
                                                        random_state=42)

    (X_train, y_train) = cutUp(X_train, y_train)
    (X_test, y_test)   = cutUp(X_test, y_test)

    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test, y_test   = shuffle(X_test, y_test, random_state=42)

    (X_train, y_train) = (np.array(X_train), np.array(y_train))
    (X_test, y_test)   = (np.array(X_test), np.array(y_test))

    return X_train, X_test, y_train, y_test


def load_sequence(index):
    features, labels, frames = loadSequences()

    features = np.expand_dims(features[index], axis=0)
    labels   = np.expand_dims(labels[index], axis=0)

    return features, labels


def loadSequences():
    features = readFeatures(feature_path)
    labels, dictionary = readLabels(label_path)

    (features, labels, frames) = sync(features, labels)
    return features, labels, frames


def readLabels(path):
    labels = list()
    label_vector = []

    with open(path, 'r') as f:
        for line in f.readlines():
            if ":" not in line:
                if label_vector != []:
                    labels.append(np.array(label_vector))

                label_vector = []
                last_time    = 0
            else:
                values = line.split()

                if values[1] == "NaN" or values[2] == "NaN":
                    continue

                label      = values[0].strip(':')
                start_time = int(values[1])
                end_time   = int(values[2])

                label_fillers = fillLabels(label, last_time,
                                           start_time, end_time)

                label_vector.extend(label_fillers)
                last_time = end_time

    labels   = labelizeStrings(labels)

    return labels


def labelizeStrings(labels):
    flat_labels = [item for sublist in labels for item in sublist]

    le = preprocessing.LabelEncoder()
    le.fit(flat_labels)

    labels = [le.transform(entry) for entry in labels]
    labels = np.array(labels)

    return labels, le.classes_


def fillLabels(label, last_time, start_time, end_time):
    null = ["NOP"]

    label_vector = []
    label_vector.extend(null * (start_time - last_time))
    label_vector.extend([label] * (end_time - start_time))
    return label_vector


def readFeatures(path):
    # Lists entries in directory and Filters out those which are not files
    features = list()

    files = os.listdir(path)
    for entry in files:
        features.append(readFeatureFile(path, entry))

    return features


def readFeatureFile(path, entry):
    features = []
    last_frame = -1

    file_path = os.path.join(path, entry)
    with open(file_path, 'r') as f:
        for line in f.readlines():
            feature_vector = [float(value) for value in line.split()]

            # Avoid duplicate frames
            if feature_vector[0] != last_frame:
                features.append(feature_vector)

            last_frame = feature_vector[0]

    features = np.array(features)
    return features


def sync(features, labels):
    _labels   = []
    _features = []
    _frames   = []

    for label_vectors, feature_vectors in zip(labels, features):
        max_index_features     = np.max(feature_vectors[:, 0])
        max_index_label_vector = len(label_vectors)

        max_index = min(max_index_features, max_index_label_vector)

        # Sample label vector
        label_idxs = [int(idx) for idx in feature_vectors[:, 0]]
        label_idxs = [idx for idx in label_idxs if idx < max_index]

        feature_idxs = feature_vectors[:, 0] < max_index

        _labels.append(label_vectors[label_idxs])
        _features.append(feature_vectors[feature_idxs, 1:])
        _frames.append(np.where(label_idxs))

    assert(len(_labels[0]) == len(_features[0]))
    assert(len(_labels) == len(_features))

    return (_features, _labels, _frames)


def cutUp(features, labels):
    W = 50

    feature_segments = []
    label_segments   = []

    for feature_vectors, label_vectors in zip(features, labels):
        length = len(feature_vectors)

        for i in range(0, length - W - 1):
            # Filter out entries which are only NOP
            if np.count_nonzero(label_vectors[i:i + W]) < 5:
                continue

            feature_segments.append(feature_vectors[i:i + W, :])
            label_segments.append(label_vectors[i:i + W])

    feature_segments = np.array(feature_segments)
    label_segments   = np.array(label_segments)

    return (feature_segments, label_segments)
