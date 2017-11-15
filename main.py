# coding: utf-8
from __future__ import division
from optparse import OptionParser
from collections import OrderedDict
from time import time
import sys
import os
import numpy as np
import cPickle

import models
from utilities import *

import theano
import theano.tensor as T

MAX_EPOCHS = 50
MINIBATCH_SIZE = 128
L2_REG = 0.0
CLIPPING_THRESHOLD = 2.0
PATIENCE_EPOCHS = 1

def get_minibatch_opExtra(sample_directory, word_vocabulary, batch_size, sequence_length, shuffle=False, with_pauses=True, semitone_feature_names = [], reduced_punctuation=True):
	sample_file_list = os.listdir(sample_directory)

	if shuffle:
		np.random.shuffle(sample_file_list)

    #if reduced_punctuation

	pause_bins = create_pause_bins()
	semitone_bins = create_semitone_bins()

	X_batch = []
	Y_batch = []
	P_batch = []
	otherBatches = [[] for feature_name in semitone_feature_names]

	if len(sample_file_list) < batch_size:
		print("WARNING: Not enough samples in '%s'. Reduce mini-batch size to %d or use a dataset with at least %d words."%(
			file_name,
	        len(sample_file_list),
	        MINIBATCH_SIZE * sequence_length))

	for sample_filename in sample_file_list:
		sample_file = os.path.join(sample_directory, sample_filename)
		sample = read_proscript(sample_file)

		sample_wordIds = [word_vocabulary.get(w, word_vocabulary[UNK]) for w in sample['word']]
		if reduced_punctuation:
			sample_puncIds = [reducePuncCode(INV_PUNCTUATION_CODES[punc]) for punc in sample['punctuation']]
		else:
			sample_puncIds = [INV_PUNCTUATION_CODES[punc] for punc in sample['punctuation']]
		sample_pauseIds = convert_value_to_level_sequence(sample['pause_before'], pause_bins)
		sample_featuresIds = {feature_name:convert_value_to_level_sequence(sample[feature_name], semitone_bins) for feature_name in semitone_feature_names}

		X_batch.append(sample_wordIds)
		Y_batch.append(sample_puncIds[1:sequence_length])
		P_batch.append(sample_pauseIds) 

		for index, feature_name in enumerate(semitone_feature_names): 
			otherBatches[index].append(sample_featuresIds[feature_name])

		if len(X_batch) == batch_size:
			# Transpose, because the model assumes the first axis is time
			X = np.array(X_batch, dtype=np.int32).T
			Y = np.array(Y_batch, dtype=np.int32).T
			P = np.array(P_batch, dtype=np.int32).T
			otherTensors = [np.array(batch, dtype=np.int32).T for batch in otherBatches]
	        
			yield X, Y, P, otherTensors
	        
			X_batch = []
			Y_batch = []
			P_batch = []
			otherBatches = [[] for feature_name in semitone_feature_names]

def main(options):
	if checkArgument(options.model_name):
		model_name = options.model_name
	else:
		sys.exit("'Model name' (-m)missing!")

	if checkArgument(options.num_hidden):
		num_hidden = int(options.num_hidden)
	else:
		sys.exit("'Hidden layer size (-h) missing!")

	if checkArgument(options.num_hidden_params):
		num_hidden_params = int(options.num_hidden_params)
	else:
		sys.exit("Parameters hidden layer size (-o) missing!")

	if checkArgument(options.learning_rate):
		learning_rate = float(options.learning_rate)
	else:
		sys.exit("'Learning rate' (-l) missing!")

	if checkArgument(options.data_dir, isDir=True):
		data_dir = options.data_dir
		TRAINING_SAMPLES_DIR = os.path.join(data_dir, "train_samples")
		if not checkArgument(TRAINING_SAMPLES_DIR, isDir=True):
			sys.exit("TRAINING dir missing!")
		DEV_SAMPLES_DIR = os.path.join(data_dir, "dev_samples")
		if not checkArgument(DEV_SAMPLES_DIR, isDir=True):
			sys.exit("DEV dir missing!")
		WORD_VOCAB_FILE = os.path.join(data_dir, "vocabulary.pickle")
		if not checkArgument(WORD_VOCAB_FILE, isFile=True):
			sys.exit("WORD_VOCAB file missing!")
		METADATA_FILE = os.path.join(data_dir, "metadata.pickle")
		if not checkArgument(METADATA_FILE, isFile=True):
			sys.exit("METADATA file missing!")
	else:
		sys.exit("Data directory missing")

	model_file_name = "Model_single-stage_%s_h%d_lr%s.pcl"%(model_name, num_hidden, learning_rate)

	print("num_hidden:%i, learning rate:%.2f, model filename:%s"%(num_hidden, learning_rate, model_file_name))

	#Read corpus metadata
	with open(METADATA_FILE, 'rb') as f:
		metadata = cPickle.load(f)

	x_vocabulary_size = metadata['word_vocab_size']
	print("Word vocabulary size:%i"%x_vocabulary_size)
	if options.reduced_punctuation:
		y_vocabulary_size = metadata['punc_red_vocab_size']
		print("Using reduced punctuation set. (Size:%i)"%y_vocabulary_size)
	else:
		y_vocabulary_size = metadata['punc_vocab_size']
		print("Using full punctuation set. (Size:%i)"%y_vocabulary_size)

	num_pause = metadata['no_of_pause_levels'] + 1
	print("Number of pause levels with 0: %i)"%num_pause)
	num_semitone = metadata['no_of_semitone_levels'] + 1
	print("Number of semitone levels with 0: %i)"%num_semitone)

	data_sequence_length = metadata['sequence_length']

	word_vocabulary = read_vocabulary(WORD_VOCAB_FILE)

	x = T.imatrix('x')
	y = T.imatrix('y')
	lr = T.scalar('lr')
	p = None
	a = None
	b = None
	c = None

	if options.train_with_pause:
		print("Training with pause (Num of levels: %i)"%num_pause)
		p = T.imatrix('p')
	else:
		num_semitone_features = -1

	semitone_feature_names = options.semitone_features
	num_semitone_features = len(options.semitone_features)

	print("Semitone features:")
	print(semitone_feature_names)

	if num_semitone_features == 1: 
		print("Training with %s (Num of levels: %i)"%(options.semitone_features[0], num_semitone))
		a = T.imatrix('a')
	elif num_semitone_features == 2:
		print("Training with %s (Num of levels: %i)"%(options.semitone_features[0], num_semitone))
		a = T.imatrix('a')
		print("Training with %s (Num of levels: %i)"%(options.semitone_features[1], num_semitone))
		b = T.imatrix('b')
	elif num_semitone_features == 3: 
		print("Training with %s (Num of levels: %i)"%(options.semitone_features[0], num_semitone))
		a = T.imatrix('a')
		print("Training with %s (Num of levels: %i)"%(options.semitone_features[1], num_semitone))
		b = T.imatrix('b')
		print("Training with %s (Num of levels: %i)"%(options.semitone_features[2], num_semitone))
		c = T.imatrix('c')
	elif num_semitone_features > 3:
		sys.exit("Too many extra features (for now)")

	continue_with_previous = False
	if os.path.isfile(model_file_name):
		while True:
			resp = raw_input("Found an existing model with the name %s. Do you want to:\n[c]ontinue training the existing model?\n[r]eplace the existing model and train a new one?\n[e]xit?\n>" % model_file_name)
			resp = resp.lower().strip()
			if resp not in ('c', 'r', 'e'):
				continue
			if resp == 'e':
				sys.exit()
			elif resp == 'c':
				continue_with_previous = True
			break

	if continue_with_previous:
		print "Loading previous model state" 

		net, state = models.load(model_file_name, MINIBATCH_SIZE, x=x, p=p, feature_a=a, feature_b=b, feature_c=c, num_semitone_features=num_semitone_features)

		gsums, learning_rate, validation_ppl_history, starting_epoch, rng = state
		best_ppl = min(validation_ppl_history)
	else:
		rng = np.random
		rng.seed(1)

		if not options.train_with_pause:
			print("Building model: GRU_single_concat_early_noPause")
			net = models.GRU_single_concat_early_noPause(
	            rng=rng,
	            x=x,
	            minibatch_size=MINIBATCH_SIZE,
	            n_hidden=num_hidden,
	            x_vocabulary_size=x_vocabulary_size,
	            y_vocabulary_size=y_vocabulary_size,
	            no_pause_levels=num_pause,
	            no_semitone_levels=num_semitone,
	        )
		elif options.train_with_pause and num_semitone_features == 0:
			print("Building model: GRU_single_concat_early")
			net = models.GRU_single_concat_early(
	            rng=rng,
	            x=x,
	            minibatch_size=MINIBATCH_SIZE,
	            n_hidden=num_hidden,
	            n_hidden_params=num_hidden_params,
	            x_vocabulary_size=x_vocabulary_size,
	            y_vocabulary_size=y_vocabulary_size,
	            no_pause_levels=num_pause,
	            no_semitone_levels=num_semitone,
	            p=p 
	        )
		elif options.train_with_pause and num_semitone_features == 1:
			print("Building model: GRU_single_concat_early_plus")
			net = models.GRU_single_concat_early_plus(
	            rng=rng,
	            x=x,
	            minibatch_size=MINIBATCH_SIZE,
	            n_hidden=num_hidden,
	            n_hidden_params=num_hidden_params,
	            x_vocabulary_size=x_vocabulary_size,
	            y_vocabulary_size=y_vocabulary_size,
	            no_pause_levels=num_pause,
	            no_semitone_levels=num_semitone,
	            p=p,
	            feature_a=a   
	        )
		elif options.train_with_pause and num_semitone_features == 2:
			print("Building model: GRU_single_concat_early_plus_plus")
			net = models.GRU_single_concat_early_plus_plus(
	            rng=rng,
	            x=x,
	            minibatch_size=MINIBATCH_SIZE,
	            n_hidden=num_hidden,
	            n_hidden_params=num_hidden_params,
	            x_vocabulary_size=x_vocabulary_size,
	            y_vocabulary_size=y_vocabulary_size,
	            no_pause_levels=num_pause,
	            no_semitone_levels=num_semitone,
	            p=p,
	            feature_a=a,
	            feature_b=b
	        )
		elif options.train_with_pause and num_semitone_features == 3:
			print("Building model: GRU_single_concat_early_plus_plus_plus")
			net = models.GRU_single_concat_early_plus_plus_plus(
	            rng=rng,
	            x=x,
	            minibatch_size=MINIBATCH_SIZE,
	            n_hidden=num_hidden,
	            n_hidden_params=num_hidden_params,
	            x_vocabulary_size=x_vocabulary_size,
	            y_vocabulary_size=y_vocabulary_size,
	            no_pause_levels=num_pause,
	            no_semitone_levels=num_semitone,
	            p=p,
	            feature_a=a,
	            feature_b=b,
	            feature_c=c
	        )
		else:
			sys.exit("Check pause levels (-p) or semitone levels (-s) parameters")

		starting_epoch = 0
		best_ppl = np.inf
		validation_ppl_history = []

		gsums = [theano.shared(np.zeros_like(param.get_value(borrow=True))) for param in net.params]

	#assign inputs

	training_inputs = [x] + [i for i in [p,a,b,c] if not i == None] + [y, lr]
	validation_inputs = [x] + [i for i in [p,a,b,c] if not i == None] + [y]

	#determine cost function
	cost = net.cost(y) + L2_REG * net.L2_sqr

	gparams = T.grad(cost, net.params)
	updates = OrderedDict()

	# Compute norm of gradients
	norm = T.sqrt(T.sum(
	           [T.sum(gparam ** 2) for gparam in gparams]
	       ))

	# Adagrad: "Adaptive subgradient methods for online learning and stochastic optimization" (2011)
	for gparam, param, gsum in zip(gparams, net.params, gsums):
		gparam = T.switch(
			T.ge(norm, CLIPPING_THRESHOLD),
	        gparam / norm * CLIPPING_THRESHOLD,
	        gparam
	    ) # Clipping of gradients
		updates[gsum] = gsum + (gparam ** 2)
		updates[param] = param - lr * (gparam / (T.sqrt(updates[gsum] + 1e-6)))

	train_model = theano.function(
		inputs=training_inputs,
		outputs=cost,
		updates=updates,
		on_unused_input='warn'
	)

	validate_model = theano.function(
	    inputs=validation_inputs,
	    outputs=net.cost(y),
	    on_unused_input='warn'
	)

	print("Training...")
	for epoch in range(starting_epoch, MAX_EPOCHS):
		t0 = time()
		total_neg_log_likelihood = 0
		total_num_output_samples = 0
		iteration = 0 
		for X, Y, P, EXTRA_FEATURE_TENSORS in get_minibatch_opExtra(TRAINING_SAMPLES_DIR, word_vocabulary, MINIBATCH_SIZE, data_sequence_length, shuffle=True, with_pauses=True, semitone_feature_names=semitone_feature_names, reduced_punctuation=options.reduced_punctuation):   #shuffle=True
			if not options.train_with_pause:
				total_neg_log_likelihood += train_model(X, Y, learning_rate)
			elif num_semitone_features == 0:
				total_neg_log_likelihood += train_model(X, P, Y, learning_rate)
			elif num_semitone_features == 1:
				total_neg_log_likelihood += train_model(X, P, EXTRA_FEATURE_TENSORS[0], Y, learning_rate)
			elif num_semitone_features == 2:
				total_neg_log_likelihood += train_model(X, P, EXTRA_FEATURE_TENSORS[0], EXTRA_FEATURE_TENSORS[1], Y, learning_rate)
			elif num_semitone_features == 3:
				total_neg_log_likelihood += train_model(X, P, EXTRA_FEATURE_TENSORS[0], EXTRA_FEATURE_TENSORS[1], EXTRA_FEATURE_TENSORS[2], Y, learning_rate)
		    
		    #total_neg_log_likelihood += train_model(X, inter_P['pause'], Y, learning_rate)
			total_num_output_samples += np.prod(Y.shape)
			iteration += 1
			if iteration % 100 == 0:
				sys.stdout.write("PPL: %.4f; Speed: %.2f sps\n" % (np.exp(total_neg_log_likelihood / total_num_output_samples), total_num_output_samples / max(time() - t0, 1e-100)))
				sys.stdout.flush()
		print("Total number of training labels: %d" % total_num_output_samples)

		total_neg_log_likelihood = 0
		total_num_output_samples = 0
		for X, Y, P, EXTRA_FEATURE_TENSORS in get_minibatch_opExtra(DEV_SAMPLES_DIR, word_vocabulary, MINIBATCH_SIZE, data_sequence_length, shuffle=False, with_pauses=True, semitone_feature_names=semitone_feature_names, reduced_punctuation=options.reduced_punctuation):
			if not options.train_with_pause:
				total_neg_log_likelihood += validate_model(X, Y)
			elif num_semitone_features == 0:
				total_neg_log_likelihood += validate_model(X, P, Y)
			elif num_semitone_features == 1:
				total_neg_log_likelihood += validate_model(X, P, EXTRA_FEATURE_TENSORS[0], Y)
			elif num_semitone_features == 2:
				total_neg_log_likelihood += validate_model(X, P, EXTRA_FEATURE_TENSORS[0], EXTRA_FEATURE_TENSORS[1], Y)
			elif num_semitone_features == 3:
				total_neg_log_likelihood += validate_model(X, P, EXTRA_FEATURE_TENSORS[0], EXTRA_FEATURE_TENSORS[1], EXTRA_FEATURE_TENSORS[2], Y)
			total_num_output_samples += np.prod(Y.shape)
		print("Total number of validation labels: %d" % total_num_output_samples)

		ppl = np.exp(total_neg_log_likelihood / total_num_output_samples)
		validation_ppl_history.append(ppl)

		print("Validation perplexity is %s"%np.round(ppl, 4))

		if ppl <= best_ppl:
			best_ppl = ppl
			net.save(model_file_name, gsums=gsums, learning_rate=learning_rate, validation_ppl_history=validation_ppl_history, best_validation_ppl=best_ppl, epoch=epoch, random_state=rng.get_state())
		elif best_ppl not in validation_ppl_history[-PATIENCE_EPOCHS:]:
			print("Finished!")
			print("Best validation perplexity was %s"%best_ppl)
			break

if __name__ == "__main__":
	usage = "usage: %prog [-s infile] [option]"
	parser = OptionParser(usage=usage)
	parser.add_option("-m", "--modelname", dest="model_name", default=None, help="output model filename", type="string")
	parser.add_option("-n", "--hiddensize", dest="num_hidden", default=100, help="hidden layer size", type="string")
	parser.add_option("-o", "--paramhiddensize", dest="num_hidden_params", default=10, help="params hidden layer size", type="string")
	parser.add_option("-l", "--learningrate", dest="learning_rate", default=0.05, help="hidden layer size", type="string")
	parser.add_option("-d", "--datadir", dest="data_dir", default=None, help="Data directory with training/testing/development sets, vocabulary and corpus metadata pickle files", type="string")
	parser.add_option("-p", "--train_with_pause", dest="train_with_pause", default=False, help="train with pause", action="store_true")
	parser.add_option("-f", "--semitone_features", dest="semitone_features", default=[], help="semitone features to train with", type="string", action='append')
	parser.add_option("-r", "--reduced_punctuation", dest="reduced_punctuation", default=False, help="Use reduced punctuation vocabulary", action="store_true")

	(options, args) = parser.parse_args()
	main(options)