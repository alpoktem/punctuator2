# coding: utf-8
from __future__ import division
from optparse import OptionParser
from collections import OrderedDict
import sys
import os
import numpy as np
from time import time
import yaml

import models
from models import PuncTensor
from utilities import *

import theano
import theano.tensor as T
from theano.compile.io import In

MAX_EPOCHS = 50
L2_REG = 0.0
CLIPPING_THRESHOLD = 2.0
PATIENCE_EPOCHS = 1

def get_minibatch(sample_directory, vocabulary_dict, leveler_dict, batch_size, sequence_length, shuffle=False, input_feature_names = [], reduced_punctuation=True, output_label="punctuation_before"):
	sample_file_list = os.listdir(sample_directory)

	if shuffle:
		np.random.shuffle(sample_file_list)

	input_batches = {feature_name:[] for feature_name in input_feature_names}
	output_batch = []

	if len(sample_file_list) < batch_size:
		print("WARNING: Not enough samples in '%s'. Reduce mini-batch size to %d or use a dataset with at least %d words."%(
			file_name,
	        len(sample_file_list),
	        batch_size * sequence_length))

	for sample_filename in sample_file_list:
		proscript_file = os.path.join(sample_directory, sample_filename)
		try:
			proscript = read_proscript(proscript_file)
		except:
			print("Couldn't read %s"%proscript_file)
			continue

		input_sample_layers = {}

		for feature_name in input_feature_names:
			if feature_name in vocabulary_dict.keys():
				vocabulary = vocabulary_dict[feature_name]
				sample_layer = [vocabulary.get(w, vocabulary[UNK]) for w in proscript[feature_name]]
			elif feature_name in leveler_dict.keys():
				get_level_func = leveler_dict[feature_name]
				sample_layer = [get_level_func(v) for v in proscript[feature_name]]
				#print(proscript[feature_name])
			else:
				sample_layer = proscript[feature_name]

			input_length = len(sample_layer)
			input_sample_layers[feature_name] = sample_layer

		if reduced_punctuation:
			output_sample_layer = [reducePuncCode(INV_PUNCTUATION_CODES[punc]) for punc in proscript[output_label]]
		else:
			output_sample_layer = [INV_PUNCTUATION_CODES[punc] for punc in proscript[output_label]]

		#need padding for batch processing
		if batch_size > 1:
			for feature_name in input_sample_layers.keys():
				if feature_name in vocabulary_dict.keys():
					vocabulary = vocabulary_dict[feature_name]
					input_sample_layers[feature_name] = pad(input_sample_layers[feature_name], sequence_length, vocabulary[EMP])
				elif feature_name in leveler_dict.keys():
					get_level_func = leveler_dict[feature_name]
					input_sample_layers[feature_name] = pad(input_sample_layers[feature_name], sequence_length, get_level_func(0.0))
				else:
					input_sample_layers[feature_name] = pad(input_sample_layers[feature_name], sequence_length, 0.0)
			output_sample_layer = pad(output_sample_layer, sequence_length, INV_PUNCTUATION_CODES[EMPTY])
			input_length = sequence_length

		# print(input_sample_layers)
		# print(output_sample_layer)
		# input("...")

		#add sample to batch
		for feature_name in input_sample_layers.keys():
			input_batches[feature_name].append(input_sample_layers[feature_name])

		output_batch.append(output_sample_layer[1:input_length])

		#yield batch if batch size is reached
		if len(output_batch) == batch_size:
			input_tensors = {batch_name: np.array(input_batches[batch_name], dtype=np.int32).T for batch_name in input_batches.keys()}
			output_tensor = np.array(output_batch, dtype=np.int32).T

			yield input_tensors, output_tensor

			input_batches = {feature_name:[] for feature_name in input_feature_names}
			output_batch = []
		
def main(options):
	if checkArgument(options.params_filename, isFile=True):
		with open(options.params_filename, 'r') as ymlfile:
			config = yaml.load(ymlfile)
	else:
		sys.exit("Parameters file missing")

	if checkArgument(options.model_name):
		model_name = options.model_name
	else:
		sys.exit("'Model name' (-m)missing!")

	num_hidden = int(config["NUM_HIDDEN_OUTPUT"])
	learning_rate = float(config["LEARNING_RATE"])
	batch_size = int(config["BATCH_SIZE"])
	sample_size = int(config["SAMPLE_SIZE"])
	output_label = config["OUTPUT_LABEL"]
	data_dir = config["DATA_DIR"]
	input_feature_names = options.input_features
	vocabulary_dict = {}
	leveler_dict = {}
	no_of_levels_dict = {}

	if checkArgument(data_dir, isDir=True):
		TRAINING_SAMPLES_DIR = os.path.join(data_dir, "train_samples")
		if not checkArgument(TRAINING_SAMPLES_DIR, isDir=True):
			sys.exit("TRAINING dir missing!")
		DEV_SAMPLES_DIR = os.path.join(data_dir, "dev_samples")
		if not checkArgument(DEV_SAMPLES_DIR, isDir=True):
			sys.exit("DEV dir missing!")
	else:
		sys.exit("Data directory missing")

	if options.build_on_stage_1:
		stage1_model_file_name = options.build_on_stage_1
		model_file_name = "Model_stage-2_%s.pcl"%(model_name)
	else:
		model_file_name = "Model_single-stage_%s.pcl"%(model_name)

	#check if model with name already exists
	if checkArgument(model_file_name, isFile=True):
		answer = input("Model with same name exists. Overwrite? (y/n)")
		if not answer.lower() == "y":
			sys.exit("Change model name.")

	print("model filename:%s"%model_file_name)
	print("num_hidden:%i, learning rate:%.2f"%(num_hidden, learning_rate))
	print("batch_size:%i, sample padding length:%i"%(batch_size, sample_size))

	for feature_name in input_feature_names:
		#Load vocabularies of vocabularized features
		if feature_name in config["FEATURE_VOCABULARIES"].keys():
			VOCAB_FILE = os.path.join(data_dir, config["FEATURE_VOCABULARIES"][feature_name])
			if not checkArgument(VOCAB_FILE, isFile=True):
				sys.exit("%s vocabulary file missing!"%feature_name)
			vocabulary = read_vocabulary(VOCAB_FILE)
			vocabulary_dict[feature_name] = vocabulary
			print("%s vocabulary file: %s"%(feature_name, VOCAB_FILE))
		#Load bins of leveled features
		if config["LEVELED_FEATURES"]:
			if feature_name in config["LEVELED_FEATURES"].keys():
				LEVELS_FILE = os.path.join(data_dir, config["LEVELED_FEATURES"][feature_name])
				if not checkArgument(LEVELS_FILE, isFile=True):
					sys.exit("%s levels file missing!"%feature_name)
				get_level_func, no_of_levels = get_level_maker(LEVELS_FILE)
				leveler_dict[feature_name] = get_level_func
				no_of_levels_dict[feature_name] = no_of_levels

	if options.reduced_punctuation:
		y_vocabulary_size = len(REDUCED_PUNCTUATION_VOCABULARY)
		print("Using reduced punctuation set. (Size:%i)"%y_vocabulary_size)
	else:
		y_vocabulary_size = len(PUNCTUATION_VOCABULARY)
		print("Using full punctuation set. (Size:%i)"%y_vocabulary_size)

	#prepare the tensors
	lr = T.scalar('lr')
	y = T.imatrix('y')

	#build model
	rng = np.random
	rng.seed(1)
	input_PuncTensors = []
	if options.build_on_stage_1:
		p = T.matrix('pause_before')
		stage1_net, stage1_inputs, stage1_input_feature_names, _ = models.load(stage1_model_file_name, batch_size)
		x = stage1_inputs[0]

		vocabulary_size = len(vocabulary_dict["word"])
		x_PuncTensor = PuncTensor(name="word", tensor=x, size_hidden=config["FEATURE_NUM_HIDDEN"]["word"], size_emb=config["FEATURE_EMB_SIZE"]["word"], vocabularized=True, vocabulary_size=vocabulary_size, bidirectional=True)
		p_PuncTensor = PuncTensor(name="pause_before", tensor=p, size_hidden=config["FEATURE_NUM_HIDDEN"]["pause_before"], size_emb=1, vocabularized=False, bidirectional=False)
		input_PuncTensors.append(x_PuncTensor)
		input_PuncTensors.append(p_PuncTensor)
		net = models.GRU_stage2(rng=rng,
								y_vocabulary_size=y_vocabulary_size, 
							  	minibatch_size=batch_size, 
							  	num_hidden_output = num_hidden,
							  	x_PuncTensor=x_PuncTensor,
							  	p_PuncTensor=p_PuncTensor, 
							  	stage1_net=stage1_net,
							  	stage1_inputs=stage1_inputs,
							  	stage1_input_feature_names=stage1_input_feature_names)
	else:
		for feature_name in input_feature_names:
			stats = "Training with %s"%feature_name
			if feature_name in config["BIDIRECTIONAL_FEATURES"]:
				is_bidi = True
				stats += " (bidirectional)"
			else:
				is_bidi = False

			if feature_name in vocabulary_dict.keys():
				vocabulary = vocabulary_dict[feature_name]
				stats += " (vocabulary size: %i)"%len(vocabulary)
				stats += " (embedded size: %i)"%config["FEATURE_EMB_SIZE"][feature_name]
				tensor = T.imatrix(feature_name)
				vocabulary_size = len(vocabulary_dict[feature_name])
				feature_PuncTensor = PuncTensor(name=feature_name, tensor=tensor, size_hidden=config["FEATURE_NUM_HIDDEN"][feature_name], size_emb=config["FEATURE_EMB_SIZE"][feature_name], vocabularized=True, vocabulary_size=vocabulary_size, bidirectional=is_bidi)
			elif feature_name in leveler_dict.keys():
				#get_level_func = leveler_dict[feature_name]
				no_of_levels = no_of_levels_dict[feature_name]
				stats += " (%i levels)"%no_of_levels
				tensor = T.imatrix(feature_name)
				feature_PuncTensor = PuncTensor(name=feature_name, tensor=tensor, size_hidden=config["FEATURE_NUM_HIDDEN"][feature_name], size_emb=config["FEATURE_EMB_SIZE"][feature_name], vocabularized=True, vocabulary_size=no_of_levels, bidirectional=is_bidi)
			else:
				#continous values
				stats += " (continous)"
				tensor = T.matrix(feature_name)
				feature_PuncTensor = PuncTensor(name=feature_name, tensor=tensor, size_hidden=config["FEATURE_NUM_HIDDEN"][feature_name], size_emb=1, vocabularized=False, bidirectional=is_bidi)
			input_PuncTensors.append(feature_PuncTensor)
			print(stats)

		net = models.GRU_parallel(rng=rng, 
								  y_vocabulary_size=y_vocabulary_size, 
								  minibatch_size=batch_size, 
								  num_hidden_output = num_hidden,
								  input_tensors=input_PuncTensors)

	starting_epoch = 0
	best_ppl = np.inf
	validation_ppl_history = []

	gsums = [theano.shared(np.zeros_like(param.get_value(borrow=True))) for param in net.params]

	#assign inputs
	training_inputs = [i.tensor for i in input_PuncTensors] + [y, lr]
	validation_inputs = [i.tensor for i in input_PuncTensors] + [y]

	#determine cost function
	cost = net.cost(y) + L2_REG * net.L2_sqr

	gparams = T.grad(cost, net.params)
	updates = OrderedDict()

	# Compute norm of gradients
	norm = T.sqrt(T.sum([T.sum(gparam ** 2) for gparam in gparams]))

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
		for INPUT_BATCHES, OUTPUT_BATCH in get_minibatch(TRAINING_SAMPLES_DIR, vocabulary_dict, leveler_dict, batch_size, sample_size, shuffle=True, input_feature_names=input_feature_names, reduced_punctuation=options.reduced_punctuation, output_label="punctuation_before"):
			train_arguments = [INPUT_BATCHES[puncTensor.name] for puncTensor in input_PuncTensors] + [OUTPUT_BATCH, learning_rate]
			total_neg_log_likelihood += train_model(*train_arguments)
			total_num_output_samples += np.prod(OUTPUT_BATCH.shape)
			iteration += 1
			if iteration % 100 == 0:
				sys.stdout.write("PPL: %.4f; Speed: %.2f sps\n" % (np.exp(total_neg_log_likelihood / total_num_output_samples), total_num_output_samples / max(time() - t0, 1e-100)))
				sys.stdout.flush()
		print("Total number of training labels: %d" % total_num_output_samples)

		total_neg_log_likelihood = 0
		total_num_output_samples = 0

		for INPUT_BATCHES, OUTPUT_BATCH in get_minibatch(DEV_SAMPLES_DIR, vocabulary_dict, leveler_dict, batch_size, sample_size, shuffle=False, input_feature_names=input_feature_names, reduced_punctuation=options.reduced_punctuation, output_label="punctuation_before"):
			validate_arguments = [INPUT_BATCHES[puncTensor.name] for puncTensor in input_PuncTensors] + [OUTPUT_BATCH]
			total_neg_log_likelihood += validate_model(*validate_arguments)
			total_num_output_samples += np.prod(OUTPUT_BATCH.shape)

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
	parser.add_option("-d", "--datadir", dest="data_dir", default=None, help="Data directory with training/testing/development sets, vocabulary and corpus metadata pickle files", type="string")
	parser.add_option("-n", "--hiddensize", dest="num_hidden", default=100, help="hidden layer size", type="string")
	parser.add_option("-l", "--learningrate", dest="learning_rate", default=0.05, help="hidden layer size", type="string")
	parser.add_option("-f", "--input_features", dest="input_features", default=[], help="semitone features to train with", type="string", action='append')
	parser.add_option("-r", "--reduced_punctuation", dest="reduced_punctuation", default=True, help="Use reduced punctuation vocabulary", action="store_true")
	parser.add_option("-p", "--params_file", dest="params_filename", default=None, help="params filename", type="string")
	parser.add_option("-t", "--build_on_stage_1", dest="build_on_stage_1", default=None, help="Use two stage approach. Input stage 1 model", type="string")

	(options, args) = parser.parse_args()
	main(options)