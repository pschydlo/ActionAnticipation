import numpy as np
import sys
import _pickle as cPickle
import random

def onehot(tensor, num_labels):
    return np.eye(num_labels)[tensor]

def reconstructStructure(obj_count, X_object, X_object_shared, Y_object):
    X_object_        = []
    X_object_shared_ = []
    Y_object_        = []

    X_object_all = []

    max_obj = np.max(obj_count)

    obj_count = np.array(obj_count)
    cumsum = np.cumsum(obj_count)

    first_idx  = cumsum - obj_count
    second_idx = cumsum - 1

    for i, j in zip(first_idx, second_idx):
        obj_features = np.array(X_object[:,i:(j+1),:])
        obj_shared   = np.array(X_object_shared[:,i:(j+1),:])
        #obj_y        = np.array(Y_object)[:,i:(j+1)]

        nobj = obj_features.shape[1]
        npad = ((0,0),(0, max_obj - nobj),(0,0))

        obj_features = np.pad(obj_features, pad_width=npad, mode='constant', constant_values = 0)
        obj_shared   = np.pad(obj_shared  , pad_width=npad, mode='constant', constant_values = 0)
        #obj_y        = np.pad(obj_y  , pad_width=npad, mode='constant', constant_values = 0)

        features = np.append(obj_features, obj_shared, axis=2)
        X_object_all.append(features)

        X_object_.append(obj_features)
        X_object_shared_.append(obj_shared)
        #Y_object_.append(obj_y)

    X_object_ = np.array(X_object_)
    X_object_shared_ = np.array(X_object_shared_)
    #Y_object_ = np.array(Y_object_)

    return X_object_, X_object_shared_, Y_object_


def loadTest(index, fold, sequence_length, prediction_length):
    return load(index, fold, 'test', sequence_length, prediction_length)


def loadTrain(index, fold, sequence_length, prediction_length):
    return load(index, fold, 'train', sequence_length, prediction_length)


def loadTestSequence(index, fold, sequence_length, prediction_length, activity_idx):
    return load(index, fold, 'test', sequence_length, prediction_length, activity_idx)


def load(index, fold, name, sequence_length, prediction_length, activity_idx=-1):

    main_path = '/media/Data/paul/seq2seq' # 'E:/Universidade/Projects/Vislab/code/features_cad120_ground_truth_segmentation'
    path_to_dataset = '{1}/dataset/{0}'.format(fold,main_path)
    path_to_checkpoints = '{1}/checkpoints/{0}'.format(fold,main_path)

    data = cPickle.load(open('{1}/{2}_data_{0}.pik'.format(index,path_to_dataset, name), 'rb'), encoding='latin1')

    Y_human = data['labels_human']
    X_human = data['features_human_disjoint']
    X_human_shared = data['features_human_shared']

    print("Human shape")
    print(X_human.shape)

    Y_objects = data['labels_objects']
    X_objects = data['features_objects_disjoint']
    X_objects_shared = data['features_objects_shared']

    obj_count = data['object_count']

    activity_store = data['activity_store']

    X_human_ = np.swapaxes(np.array(X_human), 0, 1)

    (X_object_, X_object_shared_, Y_object_) = reconstructStructure(obj_count, X_objects, X_objects_shared, Y_objects)

    print("X_object_")
    print(len(X_object_))
    print(X_object_.shape)

    print("line93")

    if activity_idx != -1:
        (X_object_, X_object_shared_, Y_object_) = (X_object_[activity_idx:activity_idx+1], X_object_shared_[activity_idx:activity_idx+1], Y_object_[activity_idx:activity_idx+1])


    print("line94")

    print(len(X_object_))
    print(X_object_.shape)

    X_object_ = np.append(X_object_, X_object_shared_, axis=3)

    X_object_, X_human_, Y_human_past_, Y_human_, sequence_length = create_reference(X_object_, X_human_, Y_human, sequence_length, prediction_length, "test")

    X_object_, X_human_, Y_human_past_, Y_human_, sequence_length = filter_zeros(X_object_, X_human_, Y_human_past_, Y_human_, sequence_length)

    print("object shape")
    print(X_object_.shape)

    #if name=="train":
        #[X_object_, X_human_, Y_human_past_, Y_human_] = oversample_minority_class( [X_object_, X_human_, Y_human_past_, Y_human_], 3, [3,4,5,6,8] )

    # Split sequence in features and reference
    #X_object_ = X_object_[:,:20,:,:]
    #X_human_  = X_human_[ :,:20,:]
    #Y_object_ = Y_object_

    return (X_object_, X_human_, Y_human_past_, Y_human_, Y_object_, sequence_length)


def filter_zeros(X_object_, X_human_, Y_human_past_, Y_human_, seqlen):
    _xo  = []
    _xh  = []
    _yhp = []
    _yh  = []
    _seqlen = []

    for xo, xh, yhp, yh, s in zip(X_object_, X_human_, Y_human_past_, Y_human_, seqlen):
        if np.count_nonzero(yh) > 1:
            _xo.append(xo)
            _xh.append(xh)
            _yhp.append(yhp)
            _yh.append(yh)
            _seqlen.append(s)


    return np.array(_xo), np.array(_xh), np.array(_yhp), np.array(_yh), np.array(_seqlen)


def create_reference(X_object, X_human, Y_human, sequence_length, prediction_length, name):
    X_object_= None
    X_human_= None
    Y_human_ = None
    Y_human_past_ = None

    n_past = sequence_length
    n_pred = prediction_length

    steps = 25 - (n_past + n_pred)

    for i in range(0, steps):
        if i is 0:
            X_object_ = X_object[:,:n_past,:,:]
            X_human_  = X_human[ :,:n_past,:]

            Y_human_  = Y_human.T[:,n_past:n_past+n_pred]
            Y_human_past_ = Y_human.T[:,:n_past]
        else:
            X_object_ = np.concatenate((X_object_, X_object[:,i:i+n_past,:,:]), axis=0)
            X_human_ = np.concatenate((X_human_, X_human[ :,i:i+n_past,:]), axis=0)
            Y_human_ = np.concatenate((Y_human_, Y_human.T[:,i+n_past:i+n_past+n_pred]), axis=0)

            Y_human_past_ = np.concatenate((Y_human_past_, Y_human.T[:,i:i+n_past]), axis=0)

    _sequence_length = []

    _X_object = []
    _X_human  = []
    _Y_human  = []
    _Y_human_past = []

    for xo, xh, yh in zip(X_object, X_human, Y_human):
        sequence = sampleSubSequences(25 - prediction_length, 100, 4)
        for s in sequence:
            l = s[1] - s[0]

            padlen = 25 - l - prediction_length

            _yhp = yh[s[0]:s[1]].copy()
            _xh = xh[s[0]:s[1], :].copy()
            _xo = xo[s[0]:s[1], :, :].copy()
            _yh = yh[s[1]:(s[1] + n_pred)].copy()

            # Pad with zeros
            _yhp.resize(21, 1)
            _xh.resize(21, 790)
            _xo.resize(21, 5, 1020)
            _yh.resize(prediction_length)

            _Y_human_past.append(_yhp)
            _X_human.append(_xh)
            _X_object.append(_xo)
            _Y_human.append(_yh)
            _sequence_length.append(l)

    #print(sequence_length)
    #print(_X_human[3].shape)
    #print(_X_human[2].shape)
    #print(_X_human[1].shape)

    #print(len(_sequence_length))

    _X_human = np.array(_X_human)
    _Y_human_past = np.array(_Y_human_past)
    _Y_human = np.array(_Y_human)
    _X_object = np.array(_X_object)

    Y_human_past_ = np.expand_dims(Y_human_past_, axis=2)

    sequence_length_ = np.full((X_object_.shape[0]), sequence_length)

    if name == 'train':
        return _X_object, _X_human, _Y_human_past, _Y_human, _sequence_length
    else:
        return X_object_, X_human_, Y_human_past_, Y_human_, sequence_length_


def oversample_minority_class(arrays, label_index, minority_labels = []):
    if minority_labels == []:
        return arrays

    label_array = arrays[label_index]
    idx = (label_array[:, 0] != 0)

    for i, entry in enumerate(arrays):
        oversampled_entry = np.append(entry, entry[idx], axis=0)

        idx = (oversampled_entry[:, 0] != 0)
        oversampled_entry = np.append(oversampled_entry, oversampled_entry[idx], axis=0)

        arrays[i] = oversampled_entry

    return arrays


def sampleSubSequences(length, num_samples=1, min_len=1, max_len=10):
    max_len = min(max_len, length)
    min_len = min(min_len, max_len)
    sequence = []
    for i in range(num_samples):
        l = random.randint(min_len, max_len)
        start_idx = random.randint(0, length - l)
        end_idx = start_idx + l
        if not (start_idx, end_idx) in sequence:
            sequence.append((start_idx, end_idx))

    return sequence


if __name__ == "__main__":
    load('023895','1', 'test')
