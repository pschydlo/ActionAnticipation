import sys
import tensorflow as tf
import numpy as np

sys.path.append(r'E:\Universidade\Projects\Vislab\code')

EMBEDDING_DIM = 3
FIRST_LAYER_STATE  = 10
SECOND_LAYER_STATE = 15


class RecognitionRNN(object):
    def __init__(self, X, y, batch_size, input_drop_prob, recurr_drop_prob, output_drop_prob, parameters):
        self.X = X
        self.y = y
        self.batch_size = batch_size

        self.input_drop  = input_drop_prob
        self.recurr_drop = 1 - recurr_drop_prob
        self.output_drop = output_drop_prob

        self.parameters = parameters
        self.seqlen = parameters['seqlen']

        # Construct computation graph
        self.classify()
        self.optimize()
        self.softmax_output()

    def classify(self):
        if hasattr(self, 'logit_outputs'):
            return self.logit_outputs

        # input_dropout = tf.layers.dropout(self.X, self.input_drop, seed=42)
        # self.input_dropout = input_dropout
        # self.input = self.X
        #
        # # flatened_input = tf.reshape(input_dropout, [input_dropout.shape[0]*input_dropout.shape[1], input_dropout.shape[2]])
        #
        # #feature_size = self.X.get_shape().as_list()[-1]
        # #embedding_matrix = tf.Variable(tf.random_normal([feature_size, EMBEDDING_DIM], stddev=0.35), name="embedding_matrix", dtype=tf.float32)
        #
        # input_embedding = tf.layers.dense(input_dropout, EMBEDDING_DIM, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        # self.input_embedding = input_embedding
        #
        # # flatened_embedding = tf.matmul(flatened_input, embedding_matrix)
        # # input_embedding = input_dropout #tf.reshape(flatened_embedding, [input_dropout.shape[0], input_dropout.shape[1], input_dropout.shape[2]])
        #
        # first_layer_state_size  = FIRST_LAYER_STATE
        # second_layer_state_size = SECOND_LAYER_STATE
        #
        # with tf.variable_scope('first_layer_rnn_cell'):
        #     first_layer_cell = tf.contrib.rnn.LSTMCell(first_layer_state_size,
        #                                               forget_bias=1,
        #                                               initializer=tf.orthogonal_initializer,
        #                                               activation=tf.nn.relu,
        #                                               state_is_tuple=False)
        #
        #     #first_layer_cell = tf.nn.rnn_cell.BasicRNNCell(first_layer_state_size, activation=tf.nn.relu)
        #
        # with tf.variable_scope('second_layer_rnn_cell'):
        #     pass
        #     #second_layer_cell = tf.contrib.rnn.LSTMCell(second_layer_state_size,
        #     #                                            state_is_tuple=False)
        # #
        # # dropout_encoder_cell = tf.nn.rnn_cell.DropoutWrapper(
        # #                                 first_layer_cell,
        # #                                 state_keep_prob=self.recurr_drop,
        # #                                 variational_recurrent=True,
        # #                                 dtype=tf.float32,
        # #                                 training=False,
        # #                                 seed=32)
        #
        # with tf.variable_scope('rnn'):
        #     batch_size = self.batch_size
        #     state_size = first_layer_cell.state_size
        #
        #     initial_state = tf.zeros([batch_size, state_size])
        #     outputs, states = tf.nn.dynamic_rnn(first_layer_cell,
        #                                         input_embedding,
        #                                         initial_state=initial_state)
        #
        # self.lstm_states  = states
        # self.lstm_outputs = outputs
        #
        # with tf.variable_scope('second_layer_rnn'):
        #     pass
        #     # second_outputs, _ = tf.nn.dynamic_rnn(second_layer_cell,
        #     #                                       outputs,
        #     #                                       initial_state=initial_state)
        #
        # logit_outputs = tf.layers.dense(outputs, 8, activation=tf.nn.relu, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        #
        # self.logit_outputs = logit_outputs

        input_dropout = tf.layers.dropout(self.X, self.input_drop, seed=42)

        first_layer = tf.layers.dense(input_dropout, EMBEDDING_DIM, activation=None, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))

        first_layer_drop = tf.layers.dropout(first_layer, self.output_drop, seed=42)

        first_layer_cell = tf.contrib.rnn.LSTMCell(FIRST_LAYER_STATE,
                                                   forget_bias=1,
                                                   initializer=tf.orthogonal_initializer,
                                                   activation=tf.nn.relu,
                                                   state_is_tuple=False)

        batch_size = self.batch_size
        state_size = first_layer_cell.state_size

        initial_state = tf.zeros([batch_size, state_size])
        outputs, states = tf.nn.dynamic_rnn(first_layer_cell,
                                            first_layer_drop,
                                            initial_state=initial_state)

        second_layer = tf.layers.dense(outputs, 5, activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))

        second_layer_drop = tf.layers.dropout(first_layer, self.output_drop, seed=42)

        self.second_layer = second_layer_drop

        logit_outputs = tf.layers.dense(second_layer, 8, activation=None, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))

        self.logit_outputs = logit_outputs

        return self.logit_outputs

    def gradients(self):
        with tf.name_scope("gradients"):
            gradient_res = self.optimizer.compute_gradients(self.loss())

        return gradient_res

    def softmax_output(self):
        if hasattr(self, 'softmax_outputs'):
            return self.softmax_outputs

        self.softmax_outputs = tf.nn.softmax(self.classify(),
                                             name="softmax")
        return self.softmax_outputs

    def classification_output(self):
        if hasattr(self, 'classification_outputs'):
            return self.classification_outputs

        self.classification_outputs = tf.argmax(self.softmax_output(), axis=2)
        return self.classification_outputs

    def optimize(self, learning_rate=10e-4):
        # Optimizer
        if not hasattr(self, 'optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate)

            tv = tf.trainable_variables()
            regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])

            gvs = self.optimizer.compute_gradients(self.loss()) #+ regularization_cost)
            capped_gvs = [(tf.clip_by_norm(grad, 3), var) for grad, var in gvs]

            self.gradient_vars   = [var for grad, var in gvs]
            self.gradient_values = [grad for grad, var in gvs]

            self.gradients_norm = tf.reduce_max([tf.norm(grad) for grad, var in gvs])

            self.train = self.optimizer.apply_gradients(capped_gvs)
            # self.train = self.optimizer.minimize(self.loss())

        return self.train

    def decaying_weight(self):
        if hasattr(self, 'weight_tensor'):
            return self.weight_tensor

        weights = []
        for i in range(self.seqlen):
            weights.append(np.exp(float(i)))

        weights = np.array(weights)
        weights = weights / np.linalg.norm(weights)

        weights = np.expand_dims(weights, axis=0)

        print(weights)

        weight_tensor = tf.tile(weights, [1, self.batch_size])
        self.weight_tensor = tf.cast(weight_tensor, tf.float32)
        return self.weight_tensor

    def loss(self):
        if hasattr(self, 'cost'):
            return self.cost

        y_hat_logits = self.classify()

        # masks = tf.fill([self.batch_size, self.seqlen], 1.0)
        masks = self.decaying_weight()

        cost = tf.contrib.seq2seq.sequence_loss(logits=y_hat_logits,
                                                targets=self.y,
                                                softmax_loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits,
                                                weights=masks)

        self.cost = cost
        return self.cost

    def get_accuracy(self):
        y_hat = self.classification_output()

        y_hat = tf.cast(y_hat, dtype=tf.int32)

        correct_pred = tf.equal(y_hat, self.y)

        last_entry = correct_pred[:, -1]
        return tf.reduce_mean(tf.cast(last_entry, tf.float32))

    def get_100_accuracy(self):
        y_hat = self.classification_output()

        y_hat = tf.cast(y_hat, dtype=tf.int32)

        correct_pred = tf.equal(y_hat, self.y)

        last_entry = correct_pred[:, 100]
        return tf.reduce_mean(tf.cast(last_entry, tf.float32))
