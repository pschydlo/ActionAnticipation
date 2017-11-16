import tensorflow as tf
import numpy as np

# from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell

GREEDY = 0
BEAM = 1

class ActionAnticipation(object):
    def __init__(self, X_object, X_human, y_past, y,
                 batch_size, sequence_length, prediction_length, parameters):

        # Data Plaholders
        self.X_object = X_object
        self.X_human  = X_human
        self.y        = y
        self.y_past   = y_past

        self.prediction_length = prediction_length

        self.sequence_length_tensor = sequence_length

        self.batch_size = tf.cast(batch_size, tf.int32)

        self.parameters = parameters

        # Construct computation graph
        self.predict(mode=BEAM)
        self.predict(mode=GREEDY)
        self.optimize()

        self.train_accuracy()
        self.test_accuracy()

    def predict(self, mode=GREEDY, out_mode=0, inference=False):
        if hasattr(self, 'y_beam'):
            if mode == BEAM:
                return self.y_beam
            else:
                if inference is False:
                    return self.y_greedy_train
                else:
                    return self.y_greedy_test

        # Parameters
        self.sequence_length   = self.parameters['sequence_length']

        self.hidden_state_size = 10

        beam_width = 11
        vocab_size = 11

        with tf.name_scope("scene_representation"):
            # Embedding
            OBJ_EMBEDDING_DIM = 50

            input_dropout = tf.layers.dropout(self.X_object, 0.3, seed=42)

            obj_representation = tf.layers.dense(input_dropout, OBJ_EMBEDDING_DIM, activation=None, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))

            obj_representation = tf.layers.dropout(obj_representation, 0.3, seed=42)

            obj_representation_2 = tf.layers.dense(obj_representation, 10, activation=None, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))

            scene_representation = tf.reduce_sum(obj_representation_2, axis=2)

        with tf.name_scope("feature_vector"):
            input_sequence = tf.concat([scene_representation,
                                        self.X_human, self.y_past],
                                       axis=2, name='concat')

        EMBEDDING_DIM = 50
        input_sequence = tf.layers.dense(input_sequence, EMBEDDING_DIM, activation=None, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))

        with tf.name_scope("context"):
            (context_train, context_test, context_outputs_train, context_outputs_test) = self.instantiate_context(input_sequence)

        with tf.name_scope("decoder_cell"):
            (train_dec_cell, test_dec_cell, train_cell_state, test_cell_state) = self.instantiate_decoder_cell(context_test, context_train, context_outputs_train, context_outputs_test)

        with tf.name_scope("output_embedding_layer"):
            embedding = tf.Variable(tf.random_normal([vocab_size, vocab_size], stddev=0.35), name="embedding_matrix", dtype=tf.float32)

        with tf.name_scope("output_projection"):
            output_layer = layers_core.Dense(vocab_size, use_bias=True, activation=None)

        with tf.name_scope("beam_decoder"):
            self.y_beam = self.beam_decoder_model(context_test, test_dec_cell, embedding, output_layer, beam_width)

        with tf.name_scope("greedy_decoder"):
            self.y_greedy_train, self.y_greedy_test = self.greedy_decoder_model(context_train, context_test, train_dec_cell, test_dec_cell, embedding, output_layer)

        if mode == BEAM:
            return self.y_beam
        else:
            if inference is False:
                return self.y_greedy_train
            else:
                return self.y_greedy_test

    def context_outputs_train_op(self):
        return self.context_outputs_train

    def greedy_predict(self):
        y_hat = self.predict(mode=GREEDY, inference=True)
        y_hat = tf.argmax(y_hat, axis=2, name="greedy_predict")
        return y_hat

    def beam_predict(self):
        y_hat = self.predict(mode=BEAM, inference=True)
        y_hat = tf.identity(y_hat, name="beam_predict")
        return y_hat

    def optimize(self, learning_rate=10e-4):
        # Optimizer
        if not hasattr(self, 'optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate)

            tv = tf.trainable_variables()
            regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])

            self.train     = self.optimizer.minimize(self.loss() + 0.001*regularization_cost)

            #
            # gvs = self.optimizer.compute_gradients(self.loss()) #+ regularization_cost)
            # capped_gvs = [(tf.clip_by_norm(grad, 3), var) for grad, var in gvs]
            #
            # self.gradient_vars   = [var for grad, var in gvs]
            # self.gradient_values = [grad for grad, var in gvs]
            #
            # self.gradients_norm = tf.reduce_max([tf.norm(grad) for grad, var in gvs])
            #
            # self.train = self.optimizer.apply_gradients(capped_gvs)
            # self.train = self.optimizer.minimize(self.loss())

        return self.train

    def gradient(self):
        grads = self.optimizer.compute_gradients(self.loss())
        return grads

    def encoder_model(self, input_sequence, encoder_cell):
        with tf.variable_scope('encoder'):
            initial_state = tf.zeros([self.batch_size,
                                      encoder_cell.state_size])

            sequence_length = self.sequence_length_tensor#tf.fill([self.batch_size], self.sequence_length)

            outputs, final_state = tf.nn.dynamic_rnn(encoder_cell,
                                                     input_sequence,
                                                     sequence_length,
                                                     initial_state)

        return final_state, outputs

    def loss(self, inference=False):
        if hasattr(self, 'cost'):
            return self.cost

        logits = self.predict(mode=GREEDY, inference=False)

        Y_seq_len = array_ops.fill([self.batch_size], self.prediction_length)
        masks = tf.sequence_mask(Y_seq_len,
                                 tf.reduce_max(Y_seq_len), dtype=tf.float32)

        cost = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=self.y,
                                                weights=masks)
        self.cost = cost
        return self.cost

    def test_loss(self):
        if hasattr(self, 'test_cost'):
            return self.test_cost

        logits = self.predict(mode=GREEDY, inference=True)

        Y_seq_len = array_ops.fill([self.batch_size], self.prediction_length)
        masks = tf.sequence_mask(Y_seq_len,
                                 tf.reduce_max(Y_seq_len), dtype=tf.float32)

        cost = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=self.y,
                                                weights=masks)
        self.test_cost = cost
        return self.test_cost

    def train_accuracy(self, index=0):
        if hasattr(self, 'train_acc'):
            return self.train_acc

        ids = self.greedy_predict()

        acc, acc_op = tf.metrics.accuracy(ids[:, index], self.y[:, index])
        self.train_acc = acc_op
        return self.train_acc

    def instantiate_decoder_cell(self, context_train, context_test, context_outputs_train, context_outputs_test):
        if hasattr(self, 'train_decoder_cell'):
            return (self.train_decoder_cell, self.test_decoder_cell,
                    self.train_cell_state, self.test_cell_state)

        # Shared decoder variables
        decoder_cell = rnn_cell.LSTMCell(self.hidden_state_size,
                                         state_is_tuple=False)

        dropout_decoder_cell = tf.nn.rnn_cell.DropoutWrapper(
                                        decoder_cell,
                                        state_keep_prob=0.95,
                                        variational_recurrent=True,
                                        dtype=tf.float32,
                                        seed=32)

        attention_mechanism_train = tf.contrib.seq2seq.BahdanauAttention(
                                    self.hidden_state_size,
                                    context_outputs_train,
                                    memory_sequence_length=None)

        attention_mechanism_inference = tf.contrib.seq2seq.LuongAttention(
                                    self.hidden_state_size,
                                    context_outputs_test,
                                    memory_sequence_length=None)

        att_drop_dec_cell = tf.contrib.seq2seq.AttentionWrapper(
                                    dropout_decoder_cell,
                                    attention_mechanism_train,
                                    attention_layer_size=self.hidden_state_size)

        att_dec_cell = tf.contrib.seq2seq.AttentionWrapper(
                                    decoder_cell,
                                    attention_mechanism_inference,
                                    attention_layer_size=self.hidden_state_size)

        attn_zero_test = att_dec_cell.zero_state(self.batch_size, dtype=tf.float32)
        initial_state_test = attn_zero_test.clone(cell_state=context_test)

        attn_zero_train = att_drop_dec_cell.zero_state(self.batch_size, dtype=tf.float32)
        initial_state_train = attn_zero_train.clone(cell_state=context_train)

        self.train_decoder_cell = dropout_decoder_cell
        self.test_decoder_cell  = decoder_cell

        self.train_cell_state = context_train
        self.test_cell_state  = context_test

        return (self.train_decoder_cell, self.test_decoder_cell,
                self.train_cell_state, self.test_cell_state)

    def instantiate_context(self, input_sequence):

        input_sequence_train = tf.nn.dropout(input_sequence, 0.7)

        # Shared encoder cell
        if hasattr(self, 'context_train'):
            return (self.context_train, self.context_test,
                    self.context_outputs_train, self.context_outputs_test)

        encoder_cell = tf.contrib.rnn.BasicLSTMCell(
                                self.hidden_state_size,
                                state_is_tuple=False)

        dropout_encoder_cell = tf.nn.rnn_cell.DropoutWrapper(
                                encoder_cell,
                                state_keep_prob=0.95,
                                variational_recurrent=True,
                                dtype=tf.float32,
                                seed=32)

        self.context_test, self.context_outputs_test = self.encoder_model(
                                    input_sequence,
                                    encoder_cell)

        self.context_train, self.context_outputs_train = self.encoder_model(
                                    input_sequence_train,
                                    dropout_encoder_cell)

        return (self.context_train, self.context_test,
                self.context_outputs_train, self.context_outputs_test)

    def test_accuracy(self, index=0):
        if hasattr(self, 'test_acc'):
            return self.test_acc

        ids = self.beam_predict()
        ids = ids[:, :, 0]

        acc, acc_op = tf.metrics.accuracy(ids[:, index], self.y[:, index])
        self.test_acc = acc_op
        return self.test_acc

    def greedy_decoder_model(self, initial_state_train, initial_state_test,
                             decoder_cell, inference_decoder_cell,
                             embedding, output_layer,
                             inference=False, mode=0):

        batch_size = self.batch_size

        if hasattr(self, 'training_logits'):
            if mode == 1:
                return self.outputs_manual_greedy

            if inference is True:
                return self.inference_logits

            return self.training_logits

        end_token   = -1
        start_tokens = tf.cast(self.y_past[:, -1], tf.int32)
        start_tokens = start_tokens[:, 0]

        # INFERENCE DECODER
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=embedding,
            start_tokens=start_tokens,
            end_token=end_token)

        inference_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=inference_decoder_cell,
            helper=inference_helper,
            initial_state=initial_state_test,
            output_layer=output_layer)

        inference_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=inference_decoder,
            impute_finished=False,
            maximum_iterations=self.prediction_length,
            output_time_major=False)

        #training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
        #    tf.nn.embedding_lookup(embedding, self.y),
        #    sequence_length=array_ops.fill([batch_size], self.prediction_length),
        #    embedding=embedding,
        #    sampling_probability=0.95,
        #    time_major=False,
        #    seed=32,
        #    scheduling_seed=32,
        #    )

        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cell,
            helper=inference_helper,
            initial_state=initial_state_train,
            output_layer=output_layer)

        # (finished, inputs, state) = inference_decoder.initialize()
        # self.outputs_manual_greedy = []
        # for i in range(0, 5):
        #     self.outputs_manual_greedy.append(state)
        #     (output, state, _, _) = inference_decoder.step(1, inputs, state)

        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=training_decoder,
            impute_finished=False,
            maximum_iterations=self.prediction_length,
            output_time_major=False)

        self.training_logits  = training_decoder_output.rnn_output
        self.inference_logits = inference_decoder_output.rnn_output

        return self.training_logits, self.inference_logits

    def beam_decoder_model(self, context, cell, embedding, output_layer,
                           beam_width, mode=0):

        if hasattr(self, 'beam_output') and mode == 0:
            return self.beam_output

        if hasattr(self, 'beam_output_scores') and mode == 1:
            return self.outputs_manual

        end_token = -1
        start_tokens = tf.cast(self.y_past[:, -1], tf.int32)
        start_tokens = start_tokens[:, 0]

        cell_state = tf.contrib.seq2seq.tile_batch(
            context, multiplier=beam_width)

        bsd = beam_search_decoder.BeamSearchDecoder(
            cell=cell,
            embedding=embedding,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=cell_state,
            beam_width=beam_width,
            output_layer=output_layer,
            length_penalty_weight=0.0)

        # (finished, inputs, state) = bsd.initialize()
        # self.outputs_manual = []
        # for i in range(0, 5):
        #     self.outputs_manual.append(state)
        #     (output, state, inputs, finished) = bsd.step(1, inputs, state)

        final_outputs, final_state, final_sequence_lengths = (
            decoder.dynamic_decode(
                bsd, output_time_major=False,
                maximum_iterations=self.prediction_length))

        beam_search_decoder_output = final_outputs.beam_search_decoder_output

        self.beam_output = beam_search_decoder_output.predicted_ids[:, :, :]
        beam_output_scores = beam_search_decoder_output.scores

        print("Added beam_scores")
        self.beam_output_scores = tf.identity(beam_output_scores, name="beam_scores")

        if mode == 0:
            return self.beam_output
        else:
            return self.outputs_manual
