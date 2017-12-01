# coding: utf-8
from __future__ import division
import sys
import os
import codecs
from optparse import OptionParser
from utilities import *
import models as models

import theano
import theano.tensor as T
import numpy as np

MAX_SEQUENCE_LENGTH = 50

def restore_unsequenced_test_data(test_data_path, word_vocabulary, predict_function, with_pause_feature, semitone_feature_names, write_groundtruth, sequence_length, output_text=None, output_pickle=None):
	proscript_data = read_proscript(test_data_path)
	pause_bins = create_pause_bins()
	semitone_bins = create_semitone_bins()

	word_sequence = proscript_data['word']
	pause_sequence = convert_value_to_level_sequence(proscript_data[PAUSE_FEATURE_NAME], pause_bins)
	otherfeatures_sequences = [convert_value_to_level_sequence(proscript_data[feature_name], semitone_bins) for feature_name in semitone_feature_names]

	i = 0
	with codecs.open(output_text, 'w', 'utf-8') as f_out:
		while True:
			subsequence_words = word_sequence[i: i + sequence_length]
			subsequence_wordIds = [word_vocabulary.get(w, word_vocabulary[UNK]) for w in subsequence_words]
			subsequence_pauses = pause_sequence[i: i + sequence_length]
			#if write_groundtruth:
			#	subsequence_gold_reduced_puncIds = test_data['punc.red.id'][i: i + sequence_length]
			#	subsequence_gold_puncIds = test_data['punc.id'][i: i + sequence_length]
			other_subsequences = [otherfeatures_sequences[feature_index][i: i + sequence_length] for feature_index in range(len(semitone_feature_names))]

			if len(subsequence_wordIds) == 0:
				break

			if not with_pause_feature:
				y = predict_function(to_array(subsequence_wordIds))
			else:
				if len(other_subsequences) == 0:
					y = predict_function(to_array(subsequence_wordIds), to_array(subsequence_pauses))
				if len(other_subsequences) == 1:
					y = predict_function(to_array(subsequence_wordIds), to_array(subsequence_pauses), to_array(other_subsequences[0]))
				if len(other_subsequences) == 2:
					y = predict_function(to_array(subsequence_wordIds), to_array(subsequence_pauses), to_array(other_subsequences[0]), to_array(other_subsequences[1]))
				if len(other_subsequences) == 3:
					y = predict_function(to_array(subsequence_wordIds), to_array(subsequence_pauses), to_array(other_subsequences[0]), to_array(other_subsequences[1]), to_array(other_subsequences[2]))
			 
			predicted_punctuation_sequence = [0] + [np.argmax(y_t.flatten()) for y_t in y]
			#print(predicted_punctuation_sequence)

			f_out.write(subsequence_words[0])

			last_eos_idx = 0
			punctuations = []
			for y_t in y:

				p_i = np.argmax(y_t.flatten())
				#punctuation = reverse_punctuation_vocabulary[p_i]
				punctuation = p_i

				punctuations.append(punctuation)

				if punctuation in EOS_PUNCTUATION_CODES:
					last_eos_idx = len(punctuations) # we intentionally want the index of next element

			if subsequence_words[-1] == END:
				step = len(subsequence_words) - 1
			elif last_eos_idx != 0:
				step = last_eos_idx
			else:
				step = len(subsequence_words) - 1

			for j in range(step):
				if options.readable_format:
					if punctuations[j] == 0:
						f_out.write(" ")
					else:
						f_out.write(PUNCTUATION_VOCABULARY[punctuations[j]] + " ")
				else:
					f_out.write(" " + PUNCTUATION_VOCABULARY[punctuations[j]] + " ")
				if j < step - 1:
					try: 
						f_out.write(subsequence_words[1+j])
					except:
						f_out.write("<unk>")
						print(subsequence_words[1+j])

			if subsequence_words[-1] == END:
				break

			i += step

def main(options):
	if checkArgument(options.model_file):
		model_file = options.model_file
		print("Model: %s"%model_file)
	else:
		sys.exit("Model file path argument missing")

	if checkArgument(options.vocabulary_file):
		WORD_VOCAB_FILE = options.vocabulary_file
	else:
		sys.exit("Vocabulary file path argument missing")

	if checkArgument(options.input_proscript, isFile=True):
		TEST_FILE = options.input_proscript
		print("Test file: %s"%TEST_FILE)
		if checkArgument(options.output):
			OUTPUT_FILE = options.output
			OUTPUT_DIR = None
			print("Output file: %s"%OUTPUT_FILE)
		else:
			sys.exit("Output file path argument missing")
	elif checkArgument(options.input_directory, isDir=True):
		TEST_DIR = options.input_directory
		print("Test directory: %s"%TEST_DIR)
		if checkArgument(options.output, isDir=True, createDir=True):
			OUTPUT_DIR = options.output
			print("Output directory: %s"%OUTPUT_DIR)
		else:
			sys.exit("Output directory path argument missing")
	else:
		sys.exit("File to punctuate is missing!")

	word_vocabulary = read_vocabulary(WORD_VOCAB_FILE)

	x = T.imatrix('x')
	p = None
	a = None
	b = None
	c = None

	semitone_feature_names = options.semitone_features
	num_semitone_features = len(semitone_feature_names)

	if options.trained_with_pause:
		print("Punctuating with pause")
		p = T.imatrix('p')
	else:
		num_semitone_features = -1

	print("Semitone features (%i):"%num_semitone_features)

	if num_semitone_features == 1: 
		print("Punctuating with %s"%(options.semitone_features[0]))
		a = T.imatrix('a')
	elif num_semitone_features == 2:
		print("Punctuating with %s"%(options.semitone_features[0]))
		a = T.imatrix('a')
		print("Punctuating with %s"%(options.semitone_features[1]))
		b = T.imatrix('b')
	elif num_semitone_features == 3: 
		print("Punctuating with %s"%(options.semitone_features[0]))
		a = T.imatrix('a')
		print("Punctuating with %s"%(options.semitone_features[1]))
		b = T.imatrix('b')
		print("Punctuating with %s"%(options.semitone_features[2]))
		c = T.imatrix('c')
	elif num_semitone_features > 3:
		sys.exit("Too many features (for now)")

	print("Loading model parameters...")
	net, _ = models.load(model_file, 1, x, p=p, feature_a=a, feature_b=b, feature_c=c, num_semitone_features=num_semitone_features)
	inputs = [x] + [i for i in [p,a,b,c] if not i == None]

	print("Building model...")
	predict = theano.function(inputs=inputs, outputs=net.y)

	print("Generating punctuation...")
	if not OUTPUT_DIR == None:
		#punctuate all proscripts in directory
		sample_file_list = os.listdir(TEST_DIR)
		for sample_filename in sample_file_list:
			if sample_filename.endswith(".csv"):
				TEST_FILE = os.path.join(TEST_DIR, sample_filename)
				TEST_FILE_BASENAME = os.path.splitext(os.path.basename(TEST_FILE))[0]
				TEST_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "%s.txt"%TEST_FILE_BASENAME)
				restore_unsequenced_test_data(TEST_FILE,
										  word_vocabulary=word_vocabulary,
										  predict_function=predict, 
										  with_pause_feature=options.trained_with_pause, 
										  semitone_feature_names=semitone_feature_names, 
										  write_groundtruth=True,
										  sequence_length=MAX_SEQUENCE_LENGTH,
										  output_text=TEST_OUTPUT_FILE)

		print("Predictions written to %s"%(OUTPUT_DIR))
	else:
		restore_unsequenced_test_data(TEST_FILE,
									  word_vocabulary=word_vocabulary,
									  predict_function=predict, 
									  with_pause_feature=options.trained_with_pause, 
									  semitone_feature_names=semitone_feature_names, 
									  write_groundtruth=True,
									  sequence_length=MAX_SEQUENCE_LENGTH,
									  output_text=OUTPUT_FILE)
		print("Predictions written to %s."%OUTPUT_FILE)

if __name__ == "__main__":
	usage = "usage: %prog [-s infile] [option]"
	parser = OptionParser(usage=usage)
	parser.add_option("-m", "--model_file", dest="model_file", default=None, help="model filename", type="string")
	parser.add_option("-v", "--vocabulary_file", dest="vocabulary_file", default=None, help="vocabulary file (pickle)", type="string")
	parser.add_option("-i", "--input_proscript", dest="input_proscript", default=None, help="input proscript file (csv)", type="string")
	parser.add_option("-d", "--input_directory", dest="input_directory", default=None, help="directory with all proscript (csv) files to punctuate", type="string")
	parser.add_option("-o", "--output", dest="output", default=None, help="output file/directory to write predictions", type="string")
	parser.add_option("-p", "--trained_with_pause", dest="trained_with_pause", default=False, help="flag if trained with pause", action="store_true")
	parser.add_option("-f", "--semitone_features", dest="semitone_features", default=[], help="semitone feature names", type="string", action='append')
	parser.add_option("-r", "--readable_format", dest="readable_format", default=False, help="flag if output is desired in human readable format", action='store_true')
	

	(options, args) = parser.parse_args()

	main(options)