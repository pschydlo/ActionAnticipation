from __future__ import print_function

import os

import tensorflow as tf
import action_anticipation_seq2seq as action_anticipation
import dataset_seq2seq as dataset
import random

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc

import matplotlib.pyplot as plt

import time

import numpy as np
import scipy.io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


log_id = int(time.time())
#log_id = random.randint(0, 2017)
print("Log ID: ", log_id)

training_epochs = 2000
folds = [1,2,3,4]

log_dir = './logs/' + str(log_id)

screen_step = 200
log_step    = 1000

num_labels  = 11

# Parameters
sequence_length   = 3
prediction_length = 4

(X_object_train, X_human_train, y_past_train, y_train, y_obj_train, label_count) = dataset.loadTrain('339833', 1, sequence_length, prediction_length)
(X_object_test, X_human_test, y_past_test, y_test, y_obj_test, label_count)  = dataset.loadTest( '339833', 1, sequence_length, prediction_length)


hist, bin_edges = np.histogram(y_train)
print(hist)

hist, bin_edges = np.histogram(y_test)
print(hist)

print(bin_edges)

print(X_object_train.shape)
print(X_human_train.shape)
print(y_past_train.shape)
print(y_train.shape)

# Data Plaholders
X_object = tf.placeholder(tf.float32, shape=[None, None, None, 1020], name="x_o")
X_human  = tf.placeholder(tf.float32, shape=[None, None, 790], name="x_h")
y        = tf.placeholder(tf.int32, shape=[None, prediction_length], name="y")
y_past   = tf.placeholder(tf.float32, shape=[None, None, 1], name="y_p")

batch_size = tf.placeholder(tf.int32, shape=(), name="batch_size")
sequence_length_tensor = tf.placeholder(tf.int32, shape=(None), name="seqlen")
prediction_length_tensor = tf.placeholder(tf.int32, shape=(), name="predlen")

params = {'label_count': num_labels,
          'sequence_length': sequence_length,
          'prediction_length': prediction_length}

actionmodel = action_anticipation.ActionAnticipation(X_object, X_human, y_past,
                                                     y, batch_size, sequence_length_tensor,
                                                     prediction_length_tensor,
                                                     params)

train_log = tf.summary.scalar("train_loss",
                              actionmodel.loss())

test_log  = tf.summary.scalar("validation_loss",
                              actionmodel.test_loss())

f1_test  = tf.placeholder(tf.float32)
f1_train = tf.placeholder(tf.float32)

f1_test_summ  = tf.summary.scalar("f1_test", f1_test)
f1_train_summ = tf.summary.scalar("f1_train", f1_train)

init_global = tf.global_variables_initializer()
init_local  = tf.local_variables_initializer()

# Launch the graph

test_loss_vec  = []
train_loss_vec = []

saver = tf.train.Saver()

with tf.Session() as sess:

    f1_scores = []

    scores = np.array([])
    labels = np.array([])

    for k in folds:
        writer = tf.summary.FileWriter(log_dir + '_' + str(k), graph=sess.graph)

        (X_object_train, X_human_train, y_past_train, y_train, y_obj_train, seqlen_train) = dataset.loadTrain('339833', k, sequence_length, prediction_length)
        (X_object_test, X_human_test, y_past_test, y_test, y_obj_test, seqlen_test)  = dataset.loadTest('339833', k, sequence_length, prediction_length)

        dict_train = {X_object: X_object_train, X_human: X_human_train,
                      y_past: y_past_train, y: y_train,
                      batch_size: y_train.shape[0],
                      sequence_length_tensor: seqlen_train,
                      prediction_length_tensor: prediction_length}

        dict_test  = {X_object: X_object_test, X_human: X_human_test,
                      y_past: y_past_test, y: y_test,
                      batch_size: y_test.shape[0],
                      sequence_length_tensor: seqlen_test,
                      prediction_length_tensor: prediction_length}

        print("Fold ", k, ": ", y_train.shape[0], " training samples, ", y_test.shape[0], " test samples.")

        sess.run([init_global, init_local])

        #res = sess.run(actionmodel.context_outputs_train_op(), feed_dict=dict_train)
        #print(res.shape)

        for i in range(0, training_epochs):
            res = sess.run(actionmodel.optimize(0.01), feed_dict=dict_train)

            if i % screen_step == 0:
                print(i)

                # logits = sess.run(actionmodel.predict(mode=action_anticipation.GREEDY), feed_dict=dict_train)
                # y_hat  = sess.run(actionmodel.greedy_predict(), feed_dict=dict_train)
                # print(logits[1, -1, :])
                # print(y_train[1, :])
                # print(y_hat[1, :])

                # res = sess.run(actionmodel.gradient(), feed_dict=dict_train)
                # grad_total = 0
                # for i in range(0, len(res)):
                #     grad_total += np.linalg.norm(res[i])
                #
                # print("Gradient mangnitude:", grad_total)

                # train_acc  = sess.run(actionmodel.train_accuracy(), feed_dict=dict_train)
                # train_y_hat = sess.run(actionmodel.train_predict(), feed_dict=dict_train)
                #
                # test_acc  = sess.run(actionmodel.test_accuracy(), feed_dict=dict_test)
                # test_y_hat = sess.run(actionmodel.predict(mode=action_anticipation.BEAM), feed_dict=dict_test)

                res_test_loss = sess.run(actionmodel.test_loss(), feed_dict=dict_test)
                res_train_loss = sess.run(actionmodel.loss(), feed_dict=dict_train)

                print("Train loss:", res_train_loss, "Test loss:", res_test_loss)

                # greedy_acc = np.array([])
                # beam_acc   = np.array([])
                # for i in range(0, prediction_length):
                #     greedy_acc = np.append(greedy_acc, sess.run(actionmodel.train_accuracy(index=i), feed_dict=dict_test))
                #     beam_acc   = np.append(beam_acc, sess.run(actionmodel.test_accuracy(index=i), feed_dict=dict_test))
                #
                # print("test_greedy_acc:", greedy_acc, "test_beam_acc:", beam_acc)

                # print("Test set f1:", f1_score(y_test[:,0], test_y_hat[:,0], average='weighted'))
                # print("Train set f1: ", f1_score(y_train[:, 0], train_y_hat[:, 0], average='weighted'))
                #
                # print("Test set accuracy: ", test_acc)
                # print("Train set accuracy: ", train_acc)
                # pass

            #test_y_hat = sess.run(actionmodel.predict(mode=action_anticipation.BEAM, batch_size=y_past_test.shape[0]), feed_dict={X_object: X_object_test, X_human: X_human_test, y_past: y_past_test, y: y_test})
            #print("test: ", test_y_hat[20:60, 1])
            #train_y_hat = sess.run(actionmodel.train_predict(batch_size=y_past_train.shape[0]), feed_dict={X_object: X_object_train, X_human: X_human_train, y_past: y_past_train, y: y_train})
            #print("train: ", train_y_hat[20:60, 1])

            if i % log_step == 0:
                # train_y_hat = sess.run(actionmodel.train_predict(), feed_dict=dict_train)
                # test_y_hat = sess.run(actionmodel.predict(mode=action_anticipation.BEAM), feed_dict=dict_test)
                #
                # test_f1  = f1_score(y_test[:,0], test_y_hat[:,0], average='weighted')
                # train_f1 = f1_score(y_train[:, 0], train_y_hat[:, 0], average='weighted')

                res_test_loss, test_loss = sess.run([actionmodel.loss(), test_log], feed_dict=dict_test)
                writer.add_summary(test_loss, i)

                res_train_loss, train_loss = sess.run([actionmodel.loss(), train_log], feed_dict=dict_train)
                writer.add_summary(train_loss, i)

                train_loss_vec.append(res_train_loss)
                test_loss_vec.append(res_test_loss)

                # train_f1_res, test_f1_res = sess.run([f1_test_summ, f1_train_summ], feed_dict={f1_test: test_f1, f1_train: train_f1})
                # writer.add_summary(train_f1_res, i)
                # writer.add_summary(test_f1_res, i)

                pass

        beam_y_hat = sess.run(actionmodel.beam_predict(),
                              feed_dict=dict_test)

        greedy_y_hat = sess.run(actionmodel.greedy_predict(),
                                feed_dict=dict_test)

        beam_scores = sess.run(actionmodel.beam_output_scores,
                               feed_dict=dict_test)

        if k == folds[0]:
            scores = beam_scores
            labels = y_test
        else:
            scores = np.concatenate((scores, beam_scores), axis=0)
            labels = np.concatenate((labels, y_test), axis=0)

        saver.save(sess, './models/' + str(log_id) + '/model', global_step=1000)

        # print(beam_scores.shape)
        #print(beam_scores[0, 1, :])
        #print(beam_scores[0, 2, :])
        #print(beam_scores[0, 3, :])
        #print(beam_scores[0, 4, :])

        #y_score = y_test

        #print(beam_scores.shape)
        #print(y_score)



        # out_beam = sess.run(actionmodel.predict(action_anticipation.BEAM,
        #                     inference=True, out_mode=1),
        #                     feed_dict=dict_test)
        #
        # out_greedy = sess.run(actionmodel.predict(action_anticipation.GREEDY,
        #                       inference=True, out_mode=1),
        #                       feed_dict=dict_test)

        # print(out_beam)
        # print(out_greedy)

        # for i in range(0, 5):
        #     print("-----")
        #     print(out_beam[i].cell_state[0, 0, :])
        #     print(out_greedy[i][0, :])
        #
        # print("Greedy:")
        # print(greedy_y_hat[1:5, :])
        # print("Beam:")
        # print(beam_y_hat[1:5, :, 0])
        # print("Reference:")
        # print(y_test[1:5, :])

        # fscores_beam = []
        # fscores_greedy = []
        # for i in range(0, prediction_length):
        #     fscore_beam = f1_score(y_test[:, -1], beam_y_hat[:, -1, 0], average='micro')
        #
        #     fscores_beam.append(f1_score(y_test[:, i], beam_y_hat[:, i, 0], average='micro'))
        #     fscores_greedy.append(f1_score(y_test[:, i], greedy_y_hat[:, i], average='micro'))
        #
        # print("greedy:")
        # print(fscores_greedy)
        # print("beam:")
        # print(fscores_beam)

        # conf_matrix = confusion_matrix(y_test[:, 0], beam_y_hat[:, 0, 0])
        # if k == 1:
        #     conf_matrixes = conf_matrix
        # else:
        #     conf_matrixes = conf_matrixes + conf_matrix
        #
        # # print(conf_matrix)

        #f1_scores.append(fscore_beam)

        # if confusion_matrix is None:
        #     confusion_matrix = utils.confusion_matrix(y_test, y_hat, num_labels)
        # else:
        #     confusion_matrix += utils.confusion_matrix(y_test, y_hat, num_labels)

    # Compute ROC curve and ROC area for each class
    fpr = list()
    tpr = list()
    roc_auc = list()
    for i in range(10):
        prob = np.exp(scores[:, 0, :]) / (np.sum(np.exp(scores[:, 0, :]), axis=1))[:, None]

        #print(prob_i)
        #print(y_score[:, 0])

        fpr_i, tpr_i, _ = roc_curve(labels[:, 0], prob[:, i], pos_label=i, drop_intermediate=False)
        roc_auc_i = auc(fpr_i, tpr_i)

        fpr.append(fpr_i)
        tpr.append(tpr_i)
        roc_auc.append(roc_auc_i)
        print("--")

    scipy.io.savemat('auc_' + str(log_id) + '.mat', dict(fpr=fpr, tpr=tpr, roc=roc_auc, labels=labels, prob=prob))

    # print(confusion_matrix.astype(int))
    # (fscore, recall, precision) = utils.confusion_matrix2fscore(confusion_matrix)
    #print("fscore beam", fscore_beam)
    #print("fscore greedy", fscore_greedy)
    #print(f1_scores)

    #print(conf_matrixes)

    #sess.run()

    # print(utils.conf-usion_matrix2fscore_saxena(confusion_matrix))

    scipy.io.savemat('loss_' + str(log_id) + '.mat', dict(test_loss=test_loss_vec, train_loss=train_loss_vec))
    print(log_id)

    #saver.save(sess, './models/' + str(log_id) + '/model', global_step=1000)
