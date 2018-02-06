# coding: utf-8
from __future__ import division
import sys
import os
import codecs
from optparse import OptionParser
from utilities import *
import themodel as models

import theano
import theano.tensor as T
import numpy as np

def restore_unsequenced_test_data(test_data_path, vocabulary_dict, predict_function, input_feature_names, sequence_length, output_text=None):
	proscript_data = read_proscript(test_data_path, add_end=True)

	i = 0
	with codecs.open(output_text, 'w', 'utf-8') as f_out:
		while True:
			subsequence_words = proscript_data['word'][i: i + sequence_length - 1]
			subsequences = {feature_name: proscript_data[feature_name][i: i + sequence_length] for feature_name in input_feature_names if not feature_name in vocabulary_dict.keys()}
			for feature_name in vocabulary_dict.keys():
				vocabulary = vocabulary_dict[feature_name]
				subsequences[feature_name] = [vocabulary.get(w, vocabulary[UNK]) for w in proscript_data[feature_name][i: i + sequence_length]]

			predict_from = [to_array(subsequences[feature_name]) for feature_name in input_feature_names]
			try:
				y = predict_function(*predict_from)
			except:
				print("fucked")
				print(subsequence_words)

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
						f_out.write(" " + PUNCTUATION_VOCABULARY[punctuations[j]] + " ")
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
		sys.exit("File or directory to punctuate is missing!")

	vocabulary_dict = {}
	if checkArgument(options.vocabulary_file):
		WORD_VOCAB_FILE = options.vocabulary_file
		word_vocabulary = read_vocabulary(WORD_VOCAB_FILE)
		vocabulary_dict['word'] = word_vocabulary
	if checkArgument(options.pos_vocabulary_file):
		POS_VOCAB_FILE = options.pos_vocabulary_file
		pos_vocabulary = read_vocabulary(POS_VOCAB_FILE)
		vocabulary_dict['pos'] = pos_vocabulary

	print("Loading model parameters...")
	if options.build_on_stage_1:
		net, inputs, input_feature_names, _ = models.load_stage2(model_file, 1, options.build_on_stage_1)
	else:
		net, inputs, input_feature_names, _ = models.load(model_file, 1)
	print("Model trained with:")
	print(input_feature_names)
	print(inputs)

	print("Building model...")
	predict = theano.function(inputs=inputs, outputs=net.y)

	print("Generating punctuation...")
	if not OUTPUT_DIR == None:
		#punctuate all proscripts in directory
		sample_file_list = os.listdir(TEST_DIR)
		sample_file_list.sort()
		for sample_filename in sample_file_list:
			if sample_filename.endswith(".csv"):
				TEST_FILE = os.path.join(TEST_DIR, sample_filename)
				TEST_FILE_BASENAME = os.path.splitext(os.path.basename(TEST_FILE))[0]
				TEST_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "%s.txt"%TEST_FILE_BASENAME)
				restore_unsequenced_test_data( TEST_FILE,
										  	   vocabulary_dict=vocabulary_dict,
										  	   predict_function=predict, 
										  	   input_feature_names=input_feature_names, 
										  	   sequence_length=options.sequence_length,
										  	   output_text=TEST_OUTPUT_FILE)
		print("Predictions written to %s"%(OUTPUT_DIR))
	else:
		restore_unsequenced_test_data( TEST_FILE,
								  	   vocabulary_dict=vocabulary_dict,
								  	   predict_function=predict, 
								  	   input_feature_names=input_feature_names, 
								  	   sequence_length=options.sequence_length,
								  	   output_text=OUTPUT_FILE)
		print("Predictions written to %s"%OUTPUT_FILE)

if __name__ == "__main__":
	usage = "usage: %prog [-s infile] [option]"
	parser = OptionParser(usage=usage)
	parser.add_option("-m", "--model_file", dest="model_file", default=None, help="model filename", type="string")
	parser.add_option("-v", "--vocabulary_file", dest="vocabulary_file", default=None, help="vocabulary file", type="string")
	parser.add_option("-x", "--pos_vocabulary_file", dest="pos_vocabulary_file", default=None, help="pos vocabulary file", type="string")
	parser.add_option("-i", "--input_proscript", dest="input_proscript", default=None, help="input proscript file (csv)", type="string")
	parser.add_option("-d", "--input_directory", dest="input_directory", default=None, help="directory with all proscript (csv) files to punctuate", type="string")
	parser.add_option("-o", "--output", dest="output", default=None, help="output file/directory to write predictions", type="string")
	parser.add_option("-r", "--readable_format", dest="readable_format", default=True, help="flag if output is desired in human readable format", action='store_true')
	parser.add_option("-s", "--sequence_length", dest="sequence_length", default=50, help="sequence length for punctuating", type="int")
	parser.add_option("-t", "--build_on_stage_1", dest="build_on_stage_1", default=None, help="Use two stage approach. Input stage 1 model", type="string")

	(options, args) = parser.parse_args()

	main(options)