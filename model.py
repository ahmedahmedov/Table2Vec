##Model File:
import numpy as np
import tensorflow as tf

## This is inspired by https://github.com/jiegzhan/multi-class-text-classification-cnn-rnn/blob/master/text_cnn_rnn.py
class TableQuerySimModel(object):
	def __init__(self, query_embedding_mat, table_embedding_mat, non_static_query,
				 non_static_table, hidden_unit_query, hidden_unit_table, query_sequence_length, table_sequence_length,
				num_classes, embedding_size,l2_reg_lambda=0.0):

		self.input_query = tf.placeholder(tf.int32, [None, query_sequence_length], name='input_query')
		self.input_table = tf.placeholder(tf.int32, [None, table_sequence_length], name='input_table')
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
		self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
		self.batch_size = tf.placeholder(tf.int32, [])
		self.pad = tf.placeholder(tf.float32, [None, 1, embedding_size, 1], name='pad')
		self.real_len = tf.placeholder(tf.int32, [None], name='real_len')

		l2_loss = tf.constant(0.0)

		with tf.device('/cpu:0'), tf.name_scope('embedding'):
			if not non_static_query:
				W_q = tf.constant(query_embedding_mat, name='W_q')
			else:
				W_q = tf.Variable(query_embedding_mat, name='W_q')
			if not non_static_table:
				W_t = tf.constant(table_embedding_mat, name='W_t')
			else:
				W_t = tf.Variable(table_embedding_mat, name='W_t')
			
			self.embedded_query = tf.nn.embedding_lookup(W_q, self.input_query)
			self.embedded_table = tf.nn.embedding_lookup(W_q, self.input_table)

		#lstm_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_unit)
		lstm_cell_query = tf.contrib.rnn.GRUCell(num_units=hidden_unit_query)
		lstm_cell_query = tf.contrib.rnn.DropoutWrapper(lstm_cell_query, output_keep_prob=self.dropout_keep_prob)

		lstm_cell_table = tf.contrib.rnn.LSTMCell(num_units=hidden_unit_table)
		lstm_cell_table = tf.contrib.rnn.DropoutWrapper(lstm_cell_table, output_keep_prob=self.dropout_keep_prob)
		

		self._initial_state_query = lstm_cell_query.zero_state(self.batch_size, tf.float32)
		self._initial_state_table = lstm_cell_table.zero_state(self.batch_size, tf.float32)

		
		outputs_query, state_query = tf.contrib.rnn.static_rnn(lstm_cell_query, self.embedded_query, initial_state=self._initial_state_query, sequence_length=self.real_len)
		outputs_table, state_table = tf.contrib.rnn.static_rnn(lstm_cell_table, self.embedded_table, initial_state=self._initial_state_table, sequence_length=self.real_len)
		output = tf.concat([outputs_query[-1], outputs_table[-1]],1)

		with tf.name_scope('output'):
			self.W = tf.Variable(tf.truncated_normal([hidden_unit_table + hidden_unit_query, num_classes], stddev=0.1), name='W')
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			self.scores = tf.nn.xw_plus_b(output, self.W, b, name='scores')
			self.predictions = tf.argmax(self.scores, 1, name='predictions')

		with tf.name_scope('loss'):
			losses = tf.nn.softmax_cross_entropy_with_logits(labels = self.input_y, logits = self.scores) #  only named arguments accepted            
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

		with tf.name_scope('accuracy'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')

		with tf.name_scope('num_correct'):
			correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.num_correct = tf.reduce_sum(tf.cast(correct, 'float'))