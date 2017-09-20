

import os
import sys
import json
import shutil
import pickle
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from model import TableQuerySimModel
from tokenizer_utilities import *
from utilities import *

BASE_DIR = 'Data'
EMBEDDING_BASE = '/home/max/DeepLearning/DeepTables/private/Keras/WordEmbeddings'
tmp_dir = 'tmp/'
GLOVE_DIR = EMBEDDING_BASE + '/Glove/'
TEXT_DATA = BASE_DIR + '/' + 'Q_T_Dev.tsv'

## Load embedding matrices and tokenizers:
with open(tmp_dir + 'table_embedding_matrix.pickle', 'rb') as input_file:
	table_embedding_matrix = np.float32(pickle.load(input_file))
with open(tmp_dir + 'query_embedding_matrix.pickle', 'rb') as input_file:
	query_embedding_matrix = np.float32(pickle.load(input_file))
with open(tmp_dir + 'table_tokenizer.pickle', 'rb') as input_file:
	table_tokenizer = pickle.load(input_file)
with open(tmp_dir + 'query_tokenizer.pickle', 'rb') as input_file:
	query_tokenizer = pickle.load(input_file)

## Calculate metrics:
MAX_QUERY_SEQUENCE_LENGTH = 30
MAX_TABLE_SEQUENCE_LENGTH = 200
MAX_TABLE_NB_WORDS = 60000
MAX_QUERY_NB_WORDS = 60000
QUERY_EMBEDDING_DIM = 300
TABLE_EMBEDDING_DIM = 300
hidden_unit_query = 100
hidden_unit_table = 100
batch_size = 512
dropout_keep_prob = .8
chunk_size = batch_size * 10

## Load Evaluation Data:
input_iterator_tables= parse_input(TEXT_DATA, chunk_size)
table_sequences = table_tokenizer.tokenize_texts(input_iterator_tables, column ='table')
input_iterator_queries= parse_input(TEXT_DATA, chunk_size)
query_sequences = query_tokenizer.tokenize_texts(input_iterator_queries, column ='query')
input_iterator_labels= parse_input(TEXT_DATA, chunk_size)
lables_sequences = get_labels(input_iterator_labels,'label')


#def predict_unseen_data():
with tf.Graph().as_default():
	session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	sess = tf.Session(config=session_conf)
	tf.set_random_seed(198609)

	with sess.as_default():
		model = TableQuerySimModel(
			query_embedding_mat = query_embedding_matrix,
			table_embedding_mat = table_embedding_matrix,
			non_static_query = False,
			non_static_table = False,
			hidden_unit_query = hidden_unit_query,
			hidden_unit_table = hidden_unit_table, 
			query_sequence_length = MAX_QUERY_SEQUENCE_LENGTH, 
			table_sequence_length = MAX_TABLE_SEQUENCE_LENGTH,
			num_classes = 2,
			l2_reg_lambda = 0,
			batch_size = batch_size,
			dropout_keep_prob = dropout_keep_prob)

		def predict_step(x_query_batch, x_table_batch, y_batch):
			feed_dict = {
				model.input_query: x_query_batch,
				model.input_table: x_table_batch,
				model.input_y: y_batch,
			}

			loss, accuracy, num_correct, predictions = sess.run(
				[model.loss, model.accuracy, model.num_correct, model.predictions], feed_dict)
			return accuracy, loss, num_correct, predictions

		#checkpoint_file = trained_dir + 'best_model.ckpt'
		saver = tf.train.Saver(tf.global_variables())
		saver.restore(sess, './Models_Q_Tables_2Mil/18-09-2017/model-16680')
		logging.critical('model has been loaded')

		while True:
			try:
				table_chunk = table_sequences.next()
				query_chunk = query_sequences.next()
				lable_chunk = lables_sequences.next()
				zipped = zip(query_chunk, table_chunk, lable_chunk)
				sum_accuracy = 0
				sum_loss = 0
				num_batches = 0
				total_samples = 0
				total_positive_samples = 0

				for batch in chunks(zipped, batch_size):
					queries_batch, tables_batch, y_batch = zip(*batch)
					accuracy, loss, num_correct, predictions = predict_step(queries_batch, tables_batch, y_batch)
					sum_accuracy += accuracy
					sum_loss += loss
					num_batches += 1
					total_samples+= batch_size
					total_positive_samples += np.sum(y_batch,0)[1]
					print 'accuracy:%f, loss:%f, num_correct:%f'% (accuracy,loss,num_correct)
			except StopIteration:
				print '[Inference] Averages: accuracy:%f, loss:%f, total samples:%d, total pos:%d' % (sum_accuracy / num_batches, sum_loss / num_batches, total_samples, total_positive_samples)
				break
		# predictions, predict_labels = [], []
		# for x_batch in batches:
		# 	batch_predictions = predict(x_batch)[0]
		# 	for batch_prediction in batch_predictions:
		# 		predictions.append(batch_prediction)
		# 		predict_labels.append(labels[batch_prediction])

		# df['PREDICTED'] = predict_labels
		# columns = sorted(df.columns, reverse=True)
		# df.to_csv(predicted_dir + 'predictions_all.csv', index=False, columns=columns, sep='|')

		# if y_test is not None:
		# 	y_test = np.array(np.argmax(y_test, axis=1))
		# 	accuracy = sum(np.array(predictions) == y_test) / float(len(y_test))
		# 	logging.critical('The prediction accuracy is: {}'.format(accuracy))

		# logging.critical('Prediction is complete, all files have been saved: {}'.format(predicted_dir))

# if __name__ == '__main__':
# 	predict_unseen_data()