import tensorflow as tf
import numpy as np
import dataset_seq2seq as dataset

import scipy.io
from sklearn import preprocessing

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc

np.set_printoptions(precision=3)
np.set_printoptions(threshold=np.nan)

# Parameters
sequence_length   = 3
prediction_length = 4

# FOLD
k = 4

(X_object_train, X_human_train, y_past_train, y_train, y_obj_train, seqlen_train) = dataset.loadTrain('339833', k, sequence_length, prediction_length)
(X_object_test, X_human_test, y_past_test, y_test, y_obj_test, seqlen_test)  = dataset.loadTest('339833', k, sequence_length, prediction_length)

# (X_object_seq, X_human_seq, y_past_seq, y_seq, y_obj_seq, seqlen_seq)  = dataset.loadTestSequence('339833', k, sequence_length, prediction_length, 27)


print(X_object_test.shape)
print(y_test.shape)

sess = tf.Session()

# 1505318583 seqlen 3 predlen 4 beam: 27%    greedy: 25%
# 1505318824 seqlen 3 predlen 4 beam: 30.65% greedy: 31.7%
# 1505319845 seqlen 3 predlen 4 beam: 33.36% greedy: 32.56%
# 1505320331 seqlen 3 predlen 4 beam: 29.61% greedy: 29.7%
# 1505320614 seqlen 3 predlen 4 beam: 32.15% greedy: 32.76%
# 1505321088 seqlen 3 predlen 4 beam: 33.23% greedy: 33.42%
# 1505321587 seqlen 3 predlen 4 beam: 35.41% greedy: 36.07%

# 1505322012 seqlen 3 predlen 5 beam: 20.83% greedy: 21.33%
# 1505322185 seqlen 3 predlen 5 beam: 28.40% greedy: 27.61%

# 1505322542 seqlen 3 predlen 6 beam: 23.29% greedy: 19.69%
# 1505323003 seqlen 3 predlen 6 beam: 23.7%  greedy: 23.8%
# 1505323850 seqlen 3 predlen 6 beam: 22.7%  greedy: 22.54%
# 1505325938 seqlen 3 predlen 6 beam: 23.62% greedy: 21.42%

# 1505326647 seqlen 3 predlen 8 beam: 20.19% greedy: 20.19%

# 1506869652 seqlen 3 predlen 4 has beam_scores

log_id = 1505321587
print(log_id)

# Bug with contrib tensorflow
dir(tf.contrib)

# Load metagraph and restore checkpoint
saver = tf.train.import_meta_graph('./models/' + str(log_id) + '/model-1000.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models/' + str(log_id) + '/'))

graph = tf.get_default_graph()
X_object = graph.get_tensor_by_name("x_o:0")
X_human  = graph.get_tensor_by_name("x_h:0")
y        = graph.get_tensor_by_name("y:0")
y_past   = graph.get_tensor_by_name("y_p:0")

sequence_length_tensor   = graph.get_tensor_by_name("seqlen:0")
prediction_length_tensor = graph.get_tensor_by_name("predlen:0")
batch_size = graph.get_tensor_by_name("batch_size:0")


# Now, access the op that you want to run.
greedy_predict = graph.get_tensor_by_name("greedy_predict:0")
beam_predict = graph.get_tensor_by_name("beam_predict:0")
# beam_score  = graph.get_tensor_by_name("beam_scores:0")

dict_test  = {X_object: X_object_test, X_human: X_human_test,
              y_past: y_past_test, y: y_test,
              batch_size: y_test.shape[0],
              sequence_length_tensor: seqlen_test,
              prediction_length_tensor: prediction_length}

# dict_seq = {X_object: X_object_seq, X_human: X_human_seq,
#               y_past: y_past_seq, y: y_seq,
#               batch_size: y_seq.shape[0],
#               sequence_length_tensor: seqlen_seq,
#               prediction_length_tensor: prediction_length}

y_greedy = sess.run(greedy_predict, dict_test)
print(y_greedy.shape)

print(y_greedy[:,0])
print(y_test[:,0])

scipy.io.savemat('confusion_matrix_data.mat', dict(y_test=y_test, y_greedy=y_greedy))

# [y_beam, scores_beam] = sess.run([beam_predict, beam_score], dict_test)
# print(y_greedy.shape)

# [y_beam] = sess.run([beam_predict], dict_test)

# print(y_beam.shape)

# Compute ROC curve and ROC area for each class
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#for i in range(11):
#    fpr[i], tpr[i], _ = roc_curve(scores_beam[:, 0, 0, i], y_score[:, 0], pos_label=i)
#    roc_auc[i] = auc(fpr[i], tpr[i])


#fscore_beam   = f1_score(y_test[:, -1], beam_y_hat[:, -1, 0], average='weighted')

# fscores_greedy = []
# fscores_beam   = []
# for i in range(0, prediction_length):
#     fscores_greedy.append(f1_score(y_test[:, i], y_greedy[:, i], average='weighted'))
#     fscores_beam.append(f1_score(y_test[:, i], y_beam[:, i, 0], average='weighted'))
#
# print("Greedy and beam, in this order")
# print(fscores_greedy)
# print(fscores_beam)

#
# y_hat = np.argmax(res, axis=-1)
# print(y_hat.shape)
#
# print(res.shape)
#
# def clip_to_zero(x):
#     idxs = np.absolute(x) < 1e-3
#     x[idxs] = 0
#
#     # x = np.array([np.exp(row) / np.sum(np.exp(row)) for row in x])
#     return x


# print(X_demo[0,:])

# print(clip_to_zero(res[0, :]))
# print(y_demo[0, :])
# print(np.argmax(res[0,:], axis=-1))

# res_softmax = sess.run(softmax_output,
#                        feed_dict=test_dict)
#

#
# print(y_hat.shape)
# print(y_test.shape)

# print("Last frame accuracy:")
# print(accuracy_score(y_test[:, -1], y_hat[:, -1]))
#
# print(y_test.shape)
#
# print("")
# print("Frame 100 accuracy:")
#
# accuracies = []
# for i in range(0, 220):
#     accuracies.append(accuracy_score(y_demo[:, i], y_hat[:, i]))
#
# print(accuracies)
# scipy.io.savemat('data.mat', dict(dist=res[0], label=labels[0]))

# print(res[0])
# print(labels[0])
# print(frames[0])

#lb = preprocessing.LabelBinarizer()
#lb.fit(y_demo[0])

#print(y_demo.shape)
#one_hot_reference = lb.transform(y_demo[0])

#print(one_hot_reference.shape)

#print(res[0, :, 1:].shape)

#plt.plot(res[0, :, 1:])
#plt.gca().set_color_cycle(None)
#plt.plot(one_hot_reference[:, 1:])
#plt.show()

# y_hats  = []
# y_demos = []
# for i in range(0, 40):
#     X_demo, y_demo = dataset.load_sequence(i)
#
#     feed_dict = {X: X_demo, y: y_demo, batch_size: 1}
#     res = sess.run(softmax_output, feed_dict)
#
#     res = np.squeeze(res, axis=0)
#     y_demo = np.squeeze(y_demo, axis=0)
#
#     y_hats.append(res)
#     y_demos.append(y_demo)
#
# y_hats = np.array(y_hats)
#
# print(y_hats.shape)
# y_hats = np.argmax(y_hats, axis=-1)
#
# y_demos = np.array(y_demos)

# print(y_hats.shape)
# print(y_demos.shape)
