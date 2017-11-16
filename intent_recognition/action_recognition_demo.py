import tensorflow as tf
import mirko_dataset as dataset
import numpy as np

import scipy.io
from sklearn import preprocessing

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import accuracy_score

np.set_printoptions(precision=3)
np.set_printoptions(threshold=np.nan)

X_demo, y_demo = dataset.load_sequence(3)

print("Shape tests")
print(X_demo.shape)
print(y_demo.shape)


DOWNSAMPLING_STEP = 1
WINDOW = 100
WINDOW_STEP = 5
FEATURE_SIZE = 42

SEQLEN = int(WINDOW / DOWNSAMPLING_STEP)

params = {'sampling': DOWNSAMPLING_STEP,
          'window': WINDOW,
          'window_step': WINDOW_STEP}

X_train, X_test, y_train, y_test = dataset.load(params)

print(X_test.shape)
print(y_test.shape)

#features, labels = dataset.load()

sess = tf.Session()

# 1505149932 84% W = 5 body features 88-92
# 1505157907 53% W = 100 body features 88-92
# 1505161736 96% W = 100 body features 88-92
# 1505227130 86% W = 100 joint and gaze features 42
# 1505232359 92% W = 100 joint and gaze features 42
# 1505242283 91% W = 100 joint and gaze features 42
# 1505243005 90% W = 100 joint and gaze features 42
# 1505243678 83% W = 100 joint
# 1505260218 81% W = 100 joint
# 1505261830 84% W = 100 joint
# 1505262609 86% W = 100 joint
# 1505341300 90% W = 100 joint

log_id = 1505343370

# Load metagraph and restore checkpoint
saver = tf.train.import_meta_graph('./models/' + str(log_id) + '/model-1000.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models/' + str(log_id) + '/'))

graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
y = graph.get_tensor_by_name("y:0")
input_drop = graph.get_tensor_by_name("input_drop_probability:0")
recurr_drop = graph.get_tensor_by_name("recurr_drop_probability:0")
output_drop = graph.get_tensor_by_name("output_drop_probability:0")
batch_size = graph.get_tensor_by_name("batch_size:0")

# Now, access the op that you want to run.
softmax_output = graph.get_tensor_by_name("softmax:0")

feed_dict = {X: X_demo, y: y_demo, batch_size: 40}

# test_dict  = {X: X_test, y: y_test,
#               batch_size: X_test.shape[0],
#               input_drop: 0,
#               recurr_drop: 0,
#               output_drop: 0}

res = sess.run(softmax_output, feed_dict)
print(res.shape)

y_hat = np.argmax(res, axis=-1)
print(y_hat.shape)

print(res.shape)

def clip_to_zero(x):
    idxs = np.absolute(x) < 1e-3
    x[idxs] = 0

    # x = np.array([np.exp(row) / np.sum(np.exp(row)) for row in x])
    return x


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
print(y_test.shape)

print("")
print("Frame 100 accuracy:")

accuracies = []
for i in range(0, 220):
    accuracies.append(accuracy_score(y_demo[:, i], y_hat[:, i]))

print(accuracies)
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
