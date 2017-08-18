import os
import pickle
import numpy as np
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from itertools import islice
import pandas as pd
import json
import requests 
import math

def get_labels(chunks, column):
	for chunk in chunks:
		yield to_categorical(np.asarray(chunk[column].tolist()))

def load_embeddings(embeddings_file_path):

	embeddings_index = {}
	f = open(embeddings_file_path)
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
	f.close()

	return embeddings_index

def generate_text_labels(texts_file_path,chunk_size = 1000):

	chunks = pd.read_csv(texts_file_path, sep='\t', names=['query', 'url','table','label'],chunksize = chunk_size)
	for chunk in chunks:
		chunk = chunk.dropna() 
		labels = chunk['label']
		queries = chunk['query']
		tables = chunk['table']
		yield {'query':queries,'table':tables,'label':labels}

def fit_tokenizer(texts, MAX_NB_WORDS):
	tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
	tokenizer.fit_on_texts(texts)
	return tokenizer

def save_object(file_path, object_to_save):
	with open(file_path, 'wb') as handle:
		pickle.dump(object_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_object(file_path):
	with open(file_path, 'rb') as handle:
		loaded_object = pickle.load(handle)

	return loaded_object

def load_model(json_file_path, weights_file_path):
	json_file = open(json_file_path, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(weights_file_path)
	loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	return loaded_model
def accuracy_per_class(y_pred, y_true, threshold = .5):
	##both ys are np arrays:
	shape = y_pred.shape
	num_matches = np.zeros(shape[1])
	total_samples = 0
	y_rounded = round_with_maximum_index(y_pred)
	for index in range(shape[1]):
		num_matches[index] = np.sum(y_rounded[:,index] == y_true[:,index])
	accuracy = num_matches / float(shape[0])
	overall_accuracy = sum(accuracy) / float(len(accuracy))
	return accuracy,overall_accuracy

def precision_per_class(y_pred, y_true, threshold = .5):
	##both ys are np arrays:
	shape = y_pred.shape
	all_predicted_positives = np.zeros(shape[1])
	true_positives = np.zeros(shape[1])
	total_samples = 0
	y_rounded = round_with_maximum_index(y_pred)
	for index in range(shape[1]):
		all_predicted_positives[index] = sum(y_rounded[:,index])
		true_positives[index] = sum(y_true[:,index] * y_rounded[:,index])

	precision = true_positives / (all_predicted_positives)
	#overall_precision = sum(true_positives) / float(sum(all_predicted_positives))
	overall_precision = sum(precision) / len(precision)
	return precision, overall_precision

def recall_per_class(y_pred, y_true, threshold = .5):
	##both ys are np arrays:
	shape = y_pred.shape
	all_true_positives = np.zeros(shape[1])
	true_positives = np.zeros(shape[1])
	total_samples = 0
	y_rounded = round_with_maximum_index(y_pred)

	for index in range(shape[1]):
		all_true_positives[index] = sum(y_true[:,index])
		true_positives[index] = sum(y_true[:,index] * y_rounded[:,index])

	recall = true_positives /(all_true_positives)
	#overall_recall = sum(true_positives) / float(sum(all_true_positives))
	overall_recall = sum(recall) / len(recall)
	return recall,overall_recall

def accuracy_per_class_threshold(y_pred, y_true, threshold = .5):
	##both ys are np arrays:
	shape = y_pred.shape
	num_matches = np.zeros(shape[1])
	total_samples = 0
	y_rounded = round_with_threshold(y_pred, threshold)
	for index in range(shape[1]):
		num_matches[index] = np.sum(y_rounded[:,index] == y_true[:,index])
	accuracy = num_matches / float(shape[0])
	overall_accuracy = sum(accuracy) / float(len(accuracy))
	return accuracy,overall_accuracy

def precision_per_class_threshold(y_pred, y_true, threshold = .5):
	##both ys are np arrays:
	shape = y_pred.shape
	all_predicted_positives = np.zeros(shape[1])
	true_positives = np.zeros(shape[1])
	total_samples = 0
	y_rounded = round_with_threshold(y_pred, threshold)
	for index in range(shape[1]):
		all_predicted_positives[index] = sum(y_rounded[:,index])
		true_positives[index] = sum(y_true[:,index] * y_rounded[:,index])

	precision = true_positives / (all_predicted_positives)
	#overall_precision = sum(true_positives) / float(sum(all_predicted_positives))
	overall_precision = sum(precision) / len(precision)
	return precision, overall_precision

def recall_per_class_threshold(y_pred, y_true, threshold = .5):
	##both ys are np arrays:
	shape = y_pred.shape
	all_true_positives = np.zeros(shape[1])
	true_positives = np.zeros(shape[1])
	total_samples = 0
	y_rounded = round_with_threshold(y_pred, threshold)

	for index in range(shape[1]):
		all_true_positives[index] = sum(y_true[:,index])
		true_positives[index] = sum(y_true[:,index] * y_rounded[:,index])

	recall = true_positives /(all_true_positives)
	#overall_recall = sum(true_positives) / float(sum(all_true_positives))
	overall_recall = sum(recall) / len(recall)
	return recall,overall_recall
def round_with_threshold(y, threshold = .5):
	y_rounded = np.zeros(y.shape)
	for index, value in np.ndenumerate(y):
		y_rounded[index] = int(value / threshold > 1)

	return y_rounded

def round_with_maximum_index(y):

	y_rounded = np.zeros(y.shape)
	for index, array in enumerate(y):
		max_index = np.argmax(array)
		y_rounded[index][max_index] = 1

	return y_rounded

def f1_per_class(y, threshold = .5):

	precision, overall_precision = precision_per_class(y, threshold)
	recall, overall_recall = recall_per_class(y, threshold)

	overall_f1 = 2 * overall_recall*overall_precision / (overall_precision + overall_recall)
	return 2 * (precision * recall) / (recall + precision) , overall_f1

def save_model_summary(model, filename):
    current_stdout = sys.stdout
    f = file(filename, 'w')
    sys.stdout = f
    model.summary()
    sys.stdout = current_stdout
    f.close()
    return filename

def get_AGI_encoder_vector(queries, url="http://agi-encoder:5001/encoder", batch_size=2000, report_interval=1):
    """
    Sequentially call encoder API at `url` with iterable `queries` returing a list of
    corresponding query vectors (as lists of numbers).

    Return value will look like:
    [
      [-0.004863850772380829, 0.8009979724884033, -0.4575059711933136, ...],
      [-0.1462211161851883, -0.9798189997673035, 0.7600159645080566, ...],
      ...
    ]

    `batch_size` sets number of queries per request. Prints report of progress once 
    per `report_interval` requests. Use report_interval=0 for no reports.
    """

    chunks = (queries[i:i+batch_size] for i in range(0, len(queries), batch_size))
    results = []
    for i, chunk in enumerate(chunks):
        request = requests.get(url, params={"queries": json.dumps(chunk)})
        #for spell correction option
        #request = requests.get(url, params={"queries": json.dumps(chunk), "spellcheck": ''}) 
        results += request.json()
        if report_interval > 0 and (i + 1) % report_interval == 0:
            print("Finished request {req_num} of {total_reqs}"
                  .format(req_num=(i + 1), total_reqs=math.ceil(len(queries) / batch_size)))
    return [(result['query'],result['vector']) for result in results]



def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
    	if (i+n) < len(l):
        	yield l[i:i + n]

def retrieve_AGI_vectors(input_file_name, output_file_name, input_headers):

    input_dfs = pd.read_csv(input_file_name, sep = '\t', names = input_headers)
    queries = input_dfs['query'].values.tolist()
    AGI_Encoded_Queries = get_AGI_encoder_vector(queries)
    queries = [tup[0] for tup in AGI_Encoded_Queries]
    vectors = [tup[1] for tup in AGI_Encoded_Queries]
    
    new_dfs = pd.DataFrame()
    queries_series = pd.Series(queries)
    vectors_series = pd.Series(vectors)
    new_dfs['vector'] = vectors_series
    new_dfs['query'] = queries_series
    new_dfs.to_pickle(output_file_name)

class BatchIterator:
    def __init__(self, filename, batch_size):
        self.infile = open(filename, "r")
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def next(self):
        new_batch = list(islice(self.infile, self.batch_size))
        if not new_batch:
			raise StopIteration()
        else:
            return new_batch

   		