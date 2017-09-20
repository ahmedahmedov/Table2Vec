
from tokenizer_utilities import *
from utilities import *
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from model import *
import time
import logging
import pickle as pks
import sys
#### Training file for table2vec

################################# Parameter Initilization############################
#####################################################################################

BASE_DIR = 'Data'
EMBEDDING_BASE = '/home/max/DeepLearning/DeepTables/private/Keras/WordEmbeddings'
MODEL_BASE = 'Models_Sample_Q_Table'
tmp_dir = 'tmp/'
GLOVE_DIR = EMBEDDING_BASE + '/Glove/'
TRAIN_DATA = BASE_DIR + '/' + 'Q_Tables_2Mil.tsv'
DEV_DATA = BASE_DIR + '/' +'Q_T_Dev.tsv'
### Model Parameters:
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
######
num_epochs = 2
starter_learning_rate = 1e-2
learning_rate_div = 2
stop_threshold = .01
decade_every = 3000
save_every = 100
show_stats_every = 10
logs_path = '/tmp/tensorflow_logs/' + MODEL_BASE + '/' + time.strftime("%d-%m-%Y")
chunk_size = batch_size * 10

################### Table and Query Dictionary and Embedding Generation ################
########################################################################################
print('Indexing word vectors.')
embeddings_index = load_embeddings(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')
input_iterator= parse_input(TRAIN_DATA, chunk_size)

# finally, vectorize the text samples into a 2D integer tensor
query_tokenizer = Tokenizer(MAX_QUERY_NB_WORDS, MAX_QUERY_SEQUENCE_LENGTH)
query_word_dict = query_tokenizer.gen_dict(input_iterator, column = 'query')
print('done creating query word_dict!')
print('Found %s unique tokens.' % len(query_word_dict))
## Reset the iterator:
input_iterator= parse_input(TRAIN_DATA, chunk_size)
table_tokenizer = Table_Tokenizer(MAX_TABLE_NB_WORDS, MAX_TABLE_SEQUENCE_LENGTH)
table_word_dict = table_tokenizer.gen_dict(input_iterator, column = 'table')
print('done creating table word_dict!')
print('Found %s unique tokens.' % len(table_word_dict))

# prepare embedding matrix
print('Preparing query embedding matrix.')
count_existent = 0
query_nb_words = min(MAX_QUERY_NB_WORDS, len(query_word_dict))
query_embedding_matrix = np.zeros((query_nb_words + 1, QUERY_EMBEDDING_DIM),dtype=np.float32)
for word, i in query_word_dict.items():
    if i > MAX_QUERY_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        count_existent +=1
        query_embedding_matrix[i] = embedding_vector

print('words in query embedding index:%d' % count_existent)

print('Preparing table embedding matrix.')
count_existent = 0
table_nb_words = min(MAX_TABLE_NB_WORDS, len(table_word_dict))
table_embedding_matrix = np.zeros((table_nb_words + 1, TABLE_EMBEDDING_DIM),dtype=np.float32)
trainable_indices = set()
for word, i in table_word_dict.items():
    if i > MAX_TABLE_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be randomly initialized and trained.
        count_existent +=1
        table_embedding_matrix[i] = embedding_vector
    else:
    	table_embedding_matrix[i] = np.float32(np.random.normal(0, 1, TABLE_EMBEDDING_DIM))
    	trainable_indices.add(i)
print('words in table embedding index:%d, words random initialized:%d' % (count_existent, len(trainable_indices)))

########################################################################################
########################################################################################

##Load Save To Save Time:
#Save:
with open(tmp_dir + 'table_embedding_matrix_Sample_Q_Table.pickle', 'wb') as outfile:
	pickle.dump(table_embedding_matrix, outfile, pickle.HIGHEST_PROTOCOL)
with open(tmp_dir + 'query_embedding_matrix_Sample_Q_Table.pickle', 'wb') as outfile:
	pickle.dump(query_embedding_matrix, outfile, pickle.HIGHEST_PROTOCOL)
with open(tmp_dir + 'table_tokenizer_Sample_Q_Table.pickle', 'wb') as outfile:
	pickle.dump(table_tokenizer, outfile, pickle.HIGHEST_PROTOCOL)
with open(tmp_dir + 'query_tokenizer_Sample_Q_Table.pickle', 'wb') as outfile:
	pickle.dump(query_tokenizer, outfile, pickle.HIGHEST_PROTOCOL)

#Load

with open(tmp_dir + 'table_embedding_matrix_Sample_Q_Table.pickle', 'rb') as input_file:
	table_embedding_matrix = np.float32(pickle.load(input_file))
with open(tmp_dir + 'query_embedding_matrix_Sample_Q_Table.pickle', 'rb') as input_file:
	query_embedding_matrix = np.float32(pickle.load(input_file))
with open(tmp_dir + 'table_tokenizer_Sample_Q_Table.pickle', 'rb') as input_file:
	table_tokenizer = pickle.load(input_file)
with open(tmp_dir + 'query_tokenizer_Sample_Q_Table.pickle', 'rb') as input_file:
	query_tokenizer = pickle.load(input_file)
#################################################################################################
########################################TRAINING#################################################
### Prepare Saving Objects:
current_time = time.strftime("%d-%m-%Y")
save_dirpath = os.getcwd() + '/' + MODEL_BASE + '/' + current_time

if not os.path.exists(save_dirpath):
    os.makedirs(save_dirpath)
else:
    counter = 2
    while(True):
        new_dirpath = save_dirpath + '_' + str(counter)
        if not os.path.exists(new_dirpath):
            os.makedirs(new_dirpath)
            save_dirpath = new_dirpath
            break
        counter += 1
save_dirpath += '/'
#################################################################################################
#################################################################################################
graph = tf.Graph()
timestamp = str(int(time.time()))

with graph.as_default():
	session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
	sess = tf.Session(config=session_conf)
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

		global_step = tf.Variable(0, name='global_step', trainable=False)
		learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decade_every, 0.5, staircase=True)
		optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9)
		grads_and_vars = optimizer.compute_gradients(model.loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		def train_step(x_query_batch, x_table_batch, y_batch):
			feed_dict = {
				model.input_query: x_query_batch,
				model.input_table: x_table_batch,
				model.input_y: y_batch,
			}
			_, step, loss, accuracy, summary = sess.run([train_op, global_step, model.loss, model.accuracy, model.summary], feed_dict)
			return _, step, loss, accuracy, summary
		def dev_step(x_query_batch, x_table_batch, y_batch):
			feed_dict = {
				model.input_query: x_query_batch,
				model.input_table: x_table_batch,
				model.input_y: y_batch,
			}
			loss, accuracy, num_correct, predictions,current_learning_rate = sess.run(
				[model.loss, model.accuracy, model.num_correct, model.predictions,optimizer._learning_rate], feed_dict)
			return accuracy, loss, num_correct, predictions, current_learning_rate

		with tf.name_scope('epoch_summary'):
			inference_accuracy = tf.Variable(0.0)
			inference_loss = tf.Variable(0.0)
			inference_lr = tf.Variable(0.0)
			inference_accuracy_summary = tf.summary.scalar("inference_accuracy", inference_accuracy)
			inference_loss_summary = tf.summary.scalar("inference_loss", inference_loss)
			inference_lr_summary = tf.summary.scalar("inference_learning_rate", inference_lr)
			epoch_summary_op = tf.summary.merge([inference_accuracy_summary,inference_loss_summary,inference_lr_summary])
		saver = tf.train.Saver(tf.all_variables(), max_to_keep = None)
		sess.run(tf.initialize_all_variables())

		# Training starts here
		best_accuracy, best_at_step = 0, 0
		prev_epoch_loss = sys.maxint
		num_batches = 0

		# Train the model with x_train and y_train
		for epoch in range(num_epochs):
			
			query_sequences_train, table_sequences_train, lables_sequences_train = get_iters(TRAIN_DATA, chunk_size,query_tokenizer,table_tokenizer)
		
			batch_index = 0
			curr_epoch_loss = 0
			total_samples = 0
			total_positive_samples = 0
			
			while True:
				try:
					table_chunk = table_sequences_train.next()
					query_chunk = query_sequences_train.next()
					lable_chunk = lables_sequences_train.next()

					train_zipped = zip(query_chunk, table_chunk, lable_chunk)
					
					for train_batch in chunks(train_zipped, batch_size):
						total_samples += batch_size
						batch_index +=1
						queries_train_batch, tables_train_batch, y_train_batch = zip(*train_batch)
						_, step, loss, accuracy, summary = train_step(queries_train_batch, tables_train_batch, y_train_batch)
						
						if batch_index > 0 and batch_index % show_stats_every == 0:
							print 'epoch:%d, batch:%d, loss:%f, accuracy:%f'% (epoch,batch_index,loss,accuracy)
						current_step = tf.train.global_step(sess, global_step)
						curr_epoch_loss += loss
						summary_writer.add_summary(summary, epoch * num_batches + batch_index)
						total_positive_samples += np.sum(y_train_batch,0)[1]

						if batch_index > 0 and batch_index % save_every == 0:
							query_sequences_dev, table_sequences_dev, lables_sequences_dev = get_iters(DEV_DATA, chunk_size,query_tokenizer,table_tokenizer)
							sum_accuracy_dev = 0
							num_dev_batches = 0
							sum_loss_dev = 0
							total_samples_dev = 0
							total_positive_samples_dev = 0

							while True:
								try:
									table_chunk_dev = table_sequences_dev.next()
									query_chunk_dev = query_sequences_dev.next()
									lable_chunk_dev = lables_sequences_dev.next()
									dev_zipped = zip(query_chunk_dev, table_chunk_dev, lable_chunk_dev)

									for dev_batch in chunks(dev_zipped, batch_size):
										queries_dev_batch, tables_dev_batch, y_dev_batch = zip(*dev_batch)
										accuracy, loss, num_correct, predictions, current_learning_rate= dev_step(queries_dev_batch, tables_dev_batch, y_dev_batch)
										sum_accuracy_dev += accuracy
										sum_loss_dev += loss
										num_dev_batches += 1
										total_samples_dev += batch_size
										total_positive_samples_dev += np.sum(y_dev_batch,0)[1]

								except StopIteration:
									break
							print '[Inference] epoch:%d, loss:%f, accuracy:%f, total samples:%d, total pos:%d' % (epoch,sum_loss_dev / num_dev_batches,sum_accuracy_dev / num_dev_batches, total_samples_dev, total_positive_samples_dev)
 							epoch_summary = sess.run(epoch_summary_op,{inference_loss:sum_loss_dev / num_dev_batches,
						 											inference_accuracy:sum_accuracy_dev / num_dev_batches,
						 											inference_lr:current_learning_rate})
 							summary_writer.add_summary(epoch_summary,epoch * num_batches + batch_index)
							## Do evaluation and save the model:
							saved_path = saver.save(sess, save_dirpath + 'model',global_step=epoch * num_batches + batch_index)
							print 'saved_path:' + saved_path
						# Evaluate the model with x_dev and y_dev
						# if current_step % params['evaluate_every'] == 0:
						# 	dev_batches = data_helper.batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1)

						# 	total_dev_correct = 0
						# 	for dev_batch in dev_batches:
						# 		x_dev_batch, y_dev_batch = zip(*dev_batch)
						# 		acc, loss, num_dev_correct, predictions = dev_step(x_dev_batch, y_dev_batch)
						# 		total_dev_correct += num_dev_correct
						# 	accuracy = float(total_dev_correct) / len(y_dev)
						# 	logging.info('Accuracy on dev set: {}'.format(accuracy))

						# 	if accuracy >= best_accuracy:
						# 		best_accuracy, best_at_step = accuracy, current_step
						# 		path = saver.save(sess, checkpoint_prefix, global_step=current_step)
						# 		logging.critical('Saved model {} at step {}'.format(path, best_at_step))
						# 		logging.critical('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))

				except StopIteration:
					break
			learning_rate = learning_rate / learning_rate_div
			num_batches = max(num_batches,batch_index)
			curr_epoch_loss = curr_epoch_loss / batch_index
			if epoch > 2 and ((prev_epoch_loss - curr_epoch_loss) / prev_epoch_loss)  < stop_threshold:
				print 'stop criteria achieved!'
				break
			prev_epoch_loss = curr_epoch_loss
			print 'Epoch Average Loss:%f, Total samples:%d, Total pos samples:%d' % (curr_epoch_loss, total_samples, total_positive_samples)
		logging.critical('Training is complete, testing the best model on x_test and y_test')
###############################################################