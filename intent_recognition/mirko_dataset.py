import numpy as np
import os
import scipy.io

from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from tensorflow.python import debug as tf_debug


dataset_path = ('/media/Data/paul/action_recognition/mirko_dataset')

DOWNSAMPLING_STEP = 1
WINDOW = 600
WINDOW_STEP = 1



def load(params):
    global DOWNSAMPLING_STEP, WINDOW, WINDOW_STEP

    if params is not None:
        DOWNSAMPLING_STEP = params['sampling']
        WINDOW = params['window']
        WINDOW_STEP = params['window_step']

    features, labels = readMat(dataset_path)

    X_train = []
    X_test  = []
    y_train = []
    y_test  = []

    for i in range(1, 7):
        mask = labels[:] == i

        features_class = features[mask]
        labels_class   = labels[mask]

        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(features_class,
                                                                    labels_class.astype(int),
                                                                    test_size=0.33,
                                                                    random_state=33)

        X_train.extend(X_train_class)
        X_test.extend(X_test_class)
        y_train.extend(y_train_class)
        y_test.extend(y_test_class)

    # (features, labels) = cutUp(features, labels)

    # X_train, X_test, y_train, y_test = train_test_split(features,
                                                        # labels.astype(int),
                                                        # test_size=0.33,
                                                        # random_state=33)

    X_train = np.array(X_train)
    X_test  = np.array(X_test)
    y_train = np.array(y_train)
    y_test  = np.array(y_test)

    (X_train, y_train) = cutUp(X_train, y_train)
    (X_test, y_test)   = cutUp(X_test, y_test)

    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test, y_test   = shuffle(X_test, y_test, random_state=42)

    #print("X_train:", X_train.shape)

    return X_train, X_test, y_train, y_test


def load_sequence(index):
    features, labels = readMat(dataset_path)

    #print(features[1, 80:])

    #print(features[1, :])

    # Get the test set sequences
    _, features, _, labels = train_test_split(features,
                                              labels.astype(int),
                                              test_size=0.33,
                                              random_state=42)


    # Find the sequence lengths
    sequence_features = features
    #print(sequence_features.shape)
    # print(seqlen)
    # print(sequence_features[:, 0])

    sequence_features = downsample(sequence_features, DOWNSAMPLING_STEP)

    labels = np.expand_dims(labels, axis=1)

    print(labels.shape)

    sequence_labels = np.repeat(labels, 220, axis=1)

    print(sequence_features.shape)
    print(sequence_labels.shape)

    return sequence_features, sequence_labels


def readMat(path):
    # Lists entries in directory and Filters out those which are not files
    features_path = path + '/merged_labeled_actions_v7.mat'

    mat = scipy.io.loadmat(features_path)

    joint_features = mat['joints'].T
    gaze_features  = mat['gaze'].T

    selected_features = np.concatenate((joint_features, gaze_features), axis=2)

    #selected_features = features[:, :, 88:92]

    flatened = np.reshape(selected_features, (selected_features.shape[0] * selected_features.shape[1], selected_features.shape[2]))

    mask = flatened == 0

    scaler = preprocessing.StandardScaler()
    scaler.fit(flatened)
    flatened = scaler.transform(flatened)

    flatened[mask] = 0


    normalized_features = np.reshape(flatened, (selected_features.shape[0], selected_features.shape[1], selected_features.shape[2]))
    labels = mat['merged_cuts'][0:120, 1]

    return normalized_features, labels


def downsample(sequence, step):
    idxs = list(range(0, sequence.shape[0], step))
    return sequence[idxs]


def cutUp(features, labels):
    W = WINDOW
    step = WINDOW_STEP
    sampling_step = DOWNSAMPLING_STEP

    window_samples = int(W / sampling_step)

    feature_segments = []
    label_segments   = []

    lengths = []

    # Find the sequence lengths
    for sequence in features:
        lengths.append(220) #np.max(np.nonzero(sequence)))

    for j, feature_vectors in enumerate(features):
        length = lengths[j]

        for i in range(0, length - W - 1, step):
            down_sampled_sequence_piece = downsample(feature_vectors[i:i + W, :], sampling_step)
            feature_segments.append(down_sampled_sequence_piece)
            label_segments.append(np.resize(labels[j], (window_samples)))

    feature_segments = np.array(feature_segments)
    label_segments   = np.array(label_segments)

    return (feature_segments, label_segments)
