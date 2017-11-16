from __future__ import print_function

import os

import tensorflow as tf

import action_recognition_rnn as model
import action_dataset as dataset
import mirko_dataset as mirko_dataset

from sklearn.metrics import accuracy_score

import random
import numpy as np
from sklearn.utils import shuffle
import time

from tensorflow.python import debug as tf_debug

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(precision=3)
np.set_printoptions(threshold=np.nan)

def clip_to_zero(x):
    idxs = np.absolute(x) < 1e-3
    x[idxs] = 0

    # x = np.array([np.exp(row) / np.sum(np.exp(row)) for row in x])
    return x

initial_time = int(time.time())

log_id = int(time.time())
print("Log ID: ", log_id)

log_dir = ('./logs/' + str(log_id))

ACTION = 1
MIRKO  = 2

DATASET = MIRKO

training_epochs = 3000
screen_step = 100
log_step    = 20

INPUT_DROP  = 0.5
RECUR_DROP  = 0.2
OUTPUT_DROP = 0.0

FOLDS = 1


if DATASET == ACTION:
    SEQLEN = 50
    FEATURE_SIZE = 60

    X_train, X_test, y_train, y_test = dataset.load()
    seq_features, seq_labels = dataset.load_sequence(1)


if DATASET == MIRKO:
    DOWNSAMPLING_STEP = 1
    WINDOW = 100
    WINDOW_STEP = 5
    FEATURE_SIZE = 42

    SEQLEN = int(WINDOW / DOWNSAMPLING_STEP)

    params = {'sampling': DOWNSAMPLING_STEP,
              'window': WINDOW,
              'window_step': WINDOW_STEP}

    X_train, X_test, y_train, y_test = mirko_dataset.load(params)
    seq_features, seq_labels = mirko_dataset.load_sequence(1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

hist, bin_edges = np.histogram(y_train[:, 1], bins=[1, 2, 3, 4, 5, 6, 7])
print(hist)

print(hist/np.sum(hist))

hist, bin_edges = np.histogram(y_test[:, 1], bins=[1, 2, 3, 4, 5, 6, 7])
print(hist)

print(y_test[0:30, 1])


# Data Plaholders
X = tf.placeholder(tf.float32, shape=[None, None, FEATURE_SIZE], name="X")
y = tf.placeholder(tf.int32, shape=[None, None], name="y")
batch_size = tf.placeholder(tf.int32, shape=(), name="batch_size")

input_drop  = tf.placeholder(tf.float32, shape=(), name="input_drop_probability")
recurr_drop = tf.placeholder(tf.float32, shape=(), name="recurr_drop_probability")
output_drop = tf.placeholder(tf.float32, shape=(), name="output_drop_probability")

params = {'seqlen': SEQLEN}
motionmodel = model.RecognitionRNN(X, y, batch_size, input_drop, recurr_drop, output_drop, params)

init_global = tf.global_variables_initializer()
init_local  = tf.local_variables_initializer()

train_dict = {X: X_train, y: y_train,
              batch_size: X_train.shape[0],
              input_drop: INPUT_DROP,
              recurr_drop: RECUR_DROP,
              output_drop: OUTPUT_DROP}

test_dict  = {X: X_test, y: y_test,
              batch_size: X_test.shape[0],
              input_drop: 0,
              recurr_drop: 0,
              output_drop: 0}

train_log = tf.summary.scalar("train_loss", motionmodel.loss())
test_log  = tf.summary.scalar("validation_loss", motionmodel.loss())

train_acc = tf.summary.scalar("train_accuracy", motionmodel.get_accuracy())
test_acc  = tf.summary.scalar("validation_accuracy", motionmodel.get_accuracy())

gradient_norm = tf.summary.scalar("max_gradient_norm", motionmodel.gradients_norm)

# Launch the graph

sess = tf.Session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)

tf.set_random_seed(42)

sess.run([init_global, init_local])
saver = tf.train.Saver()

writer = tf.summary.FileWriter(log_dir, graph=sess.graph)

for i in range(0, training_epochs):
    res = sess.run(motionmodel.optimize(),
                   feed_dict=train_dict)

    if i % screen_step == 0:
        print(i)
        train_loss = sess.run(motionmodel.loss(),
                              feed_dict=train_dict)

        test_loss = sess.run(motionmodel.loss(),
                             feed_dict=test_dict)

        print("train:", train_loss, "test:", test_loss)

        # grad_values = sess.run(motionmodel.gradient_values,
        #                        feed_dict=train_dict)
        #
        # grad_names = motionmodel.gradient_vars
        #
        # for name, value in zip(grad_names, grad_values):
        #     print(name)
        #     print(clip_to_zero(value))



        # res = sess.run(motionmodel.softmax_output(),
        #                feed_dict=test_dict)
        #
        # print(res[0, 0:20])
        # print(y_test[0, 0:20])

    if i % log_step == 0:
        res_test_loss, test_loss = sess.run([motionmodel.loss(), test_log], feed_dict=test_dict)
        writer.add_summary(test_loss, i)

        res_train_loss, train_loss = sess.run([motionmodel.loss(), train_log], feed_dict=train_dict)
        writer.add_summary(train_loss, i)

        res_test_accuracy, test_accuracy = sess.run([motionmodel.get_accuracy(), test_acc], feed_dict=test_dict)
        writer.add_summary(test_accuracy, i)

        res_train_accuracy, train_accuracy = sess.run([motionmodel.get_accuracy(), train_acc], feed_dict=train_dict)
        writer.add_summary(train_accuracy, i)

        res_gradient_norm, gradient_norm_log = sess.run([motionmodel.gradients_norm, gradient_norm], feed_dict=train_dict)
        writer.add_summary(gradient_norm_log, i)

    # Reshufle training set after epich
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    train_dict = {X: X_train, y: y_train,
                  batch_size: X_train.shape[0],
                  input_drop: INPUT_DROP,
                  recurr_drop: RECUR_DROP,
                  output_drop: OUTPUT_DROP}

seq_dict = {X: seq_features, y: seq_labels,
            batch_size: 1,
            input_drop: INPUT_DROP,
            recurr_drop: RECUR_DROP,
            output_drop: OUTPUT_DROP}

res_softmax = sess.run(motionmodel.softmax_output(),
                       feed_dict=test_dict)

y_hat = np.argmax(res_softmax, axis=-1)

res_logit_outputs = sess.run([motionmodel.classify()],
                             feed_dict=test_dict)


def clip_to_zero(x):
    idxs = np.absolute(x) < 1e-3
    x[idxs] = 0

    # x = np.array([np.exp(row) / np.sum(np.exp(row)) for row in x])
    return x


np.set_printoptions(suppress=True, linewidth=200)

# print("Input:")
# print(X_test[0:10, -1])
# print(res_input[0:10, -1])
# print("Input dropout:")
# print(clip_to_zero(res_input_dropout[0:10, -1]))
# print("Last frame input embedding:")
# print(clip_to_zero(res_embedding[0:10, -1]))
# print(y_test[0:6, -1])
# print("LSTM final state:")
# print(clip_to_zero(res_states[0:10, -1]))
# print("LSTM final output:")
# print(clip_to_zero(res_outputs[0:80, -1]))
# print("Logits:")
# print(clip_to_zero(res_logit_outputs[0:10, -1]))
# print(res_softmx[0:10, -1])
print("Softmax:")
print(res_softmax[0:50, -1])

print(y_test[0:50, -1])

print("Last frame accuracy:")
print(accuracy_score(y_test[:, -1], y_hat[:, -1]))
#
# print("Frame 100 accuracy:")
# print(accuracy_score(y_test[:, 100], y_hat[:, 100]))

saver.save(sess, './models/' + str(log_id) + '/model' , global_step=1000)

final_time = time.time()

diference = final_time - initial_time
print(diference/60, " minutes")
