# Small LSTM Network to Generate Text based on Alice in Wonderland
# to run on Google Cloud ML Engine


import argparse
import pickle # for handling the new data source
import h5py # for saving the model

from datetime import datetime # for filename conventions

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from tensorflow.python.lib.io import file_io # for better file I/O
import sys

# Create a function to allow for different training data and other options
def train_model(train_file='wonderland.txt',
				job_dir='./tmp/wonderland', **args):
	# set the logging path for ML Engine logging to Storage bucket
	logs_path = job_dir + '/logs/' + datetime.now().isoformat()
	print('Using logs_path located at {}'.format(logs_path))

	# Reading in the pickle file. Pickle works differently with Python 2 vs 3
	f = file_io.FileIO(train_file, mode='r')
	if sys.version_info < (3,):
		data = pickle.load(f)
	else:
		data = pickle.load(f, encoding='bytes')

	# load ascii text and covert to lowercase
	#filename = "wonderland.txt"
	#raw_text = open(filename).read()
	raw_text = data.lower()

	# create mapping of unique chars to integers
	chars = sorted(list(set(raw_text)))
	char_to_int = dict((c, i) for i, c in enumerate(chars))

	# summarize the loaded data
	n_chars = len(raw_text)
	n_vocab = len(chars)
	print ("Total Characters: ", n_chars)
	print ("Total Vocab: ", n_vocab)

	# prepare the dataset of input to output pairs encoded as integers
	seq_length = 100
	dataX = []
	dataY = []
	for i in range(0, n_chars - seq_length, 1):
		seq_in = raw_text[i:i + seq_length]
		seq_out = raw_text[i + seq_length]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append(char_to_int[seq_out])
	n_patterns = len(dataX)
	print ("Total Patterns: ", n_patterns)

	# reshape X to be [samples, time steps, features]
	X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

	# normalize
	X = X / float(n_vocab)

	# one hot encode the output variable
	y = np_utils.to_categorical(dataY)

	# define the LSTM model
	model = Sequential()
	model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
	model.add(Dropout(0.2))
	model.add(Dense(y.shape[1], activation='softmax'))

	model.summary()

	model.compile(loss='categorical_crossentropy', optimizer='adam')

	# define the checkpoint
	filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]

	# fit the model
	model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

	# Save the model locally
	model.save('model.h5')

if __name__ == '__main__':
	# Parse the input arguments for common Cloud ML Engine options
	parser = argparse.ArgumentParser()
	parser.add_argument(
	  '--train-file',
	  help='Cloud Storage bucket or local path to training data')
	parser.add_argument(
	  '--job-dir',
	  help='Cloud storage bucket to export the model and store temp files')
	args = parser.parse_args()
	arguments = args.__dict__
	train_model(**arguments)