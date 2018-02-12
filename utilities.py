# coding: utf-8
import numpy as np
import csv
import codecs
import os
import glob
from collections import defaultdict

SPACE = " "
EMPTY = " "
INV_PUNCTUATION_CODES = {EMPTY:0, SPACE:0, ',':1, '.':2, '?':3, '!':4, '-':5, ';':6, ':':7, '...':8, '':0}
PUNCTUATION_VOCABULARY = {0:SPACE, 1:',', 2:'.', 3:'?', 4:'!', 5:'-', 6:';', 7:':', 8:'...'}
REDUCED_PUNCTUATION_VOCABULARY = {0:SPACE, 1:',', 2:'.', 3:'?'}
REDUCED_INV_PUNCTUATION_CODES = {EMPTY:0, SPACE:0, ',':1, '.':2, '?':3, '':0}
EOS_PUNCTUATION_CODES = [2,3,4,5,6,7,8]

END = "<END>"
UNK = "<UNK>"
EMP = "<EMP>"
NA = "NA"

#PAUSE_FEATURE_NAME = 'pause_before'
#ALL_POSSIBLE_INPUT_FEATURES = {'word', 'pos', 'pause_before', 'speech_rate_norm', 'f0_mean', 'f0_range', 'i0_mean', 'i0_range'}

def pad(l, size, padding):
	if size >= len(l):
		return l + [padding] * abs((len(l)-size))
	else:
		return l[0:size]

def read_proscript(filename, add_end=False):
	columns = defaultdict(list) # each value in each column is appended to a list

	skip_columns = []
	with open(filename) as f:
		reader = csv.DictReader(f, delimiter='|') # read rows into a dictionary format
		for row in reader: # read a row as {column1: value1, column2: value2,...}
			for (k,v) in row.items(): # go over each column name and value 
				if not k in skip_columns:
					if "word" in k or "punctuation" in k or "pos" in k:
						columns[k].append(v) # append the value into the appropriate list
					else:
						try:
							columns[k].append(float(v)) # real value
						except ValueError:
							skip_columns.append(k)
		if add_end and not columns['word'][-1] == END:
			for k in columns.keys():
				if "word" in k or "pos" in k:
					columns[k].append(END)
				elif "punctuation" in k:
					columns[k].append("")
				else:
					columns[k].append(0.0)
	return columns

def checkArgument(argname, isFile=False, isDir=False, createDir=False):
	if not argname:
		return False
	else:
		if isFile and not os.path.isfile(argname):
			return False
		if isDir:
			if not os.path.isdir(argname):
				if createDir:
					print("Creating directory %s"%(argname))
					os.makedirs(argname)
				else:
					return False
	return True

def iterable_to_dict(arr):
	return dict((x.strip(), i) for (i, x) in enumerate(arr))

def read_vocabulary(file_name):
	with codecs.open(file_name, 'r', 'utf-8') as f:
		return iterable_to_dict(f.readlines())

def to_array(arr, dtype=np.int32):
	# minibatch of 1 sequence as column
	return np.array([arr], dtype=dtype).T

def create_pause_bins():
	bins = np.arange(0, 1, 0.05)
	bins = np.concatenate((bins, np.arange(1, 2, 0.1)))
	bins = np.concatenate((bins, np.arange(2, 5, 0.2)))
	bins = np.concatenate((bins, np.arange(5, 10, 0.5)))
	bins = np.concatenate((bins, np.arange(10, 20, 1)))
	return bins

def create_pause_bins9():
	bins = np.array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  2.  ,  3.  ,  4.  , 5. ])
	return bins

def create_pause_bins2():
	return [0.0, 1.14]

def create_pause_bins3():
	return [0.0, 0.2, 1.0]

def create_semitone_bins():
	bins = np.arange(-20, -10, 1)
	bins = np.concatenate((bins, np.arange(-10, -5, 0.5)))
	bins = np.concatenate((bins, np.arange(-5, 0, 0.25)))
	bins = np.concatenate((bins, np.arange(0, 5, 0.25)))
	bins = np.concatenate((bins, np.arange(5, 10, 0.5)))
	bins = np.concatenate((bins, np.arange(10, 20, 1)))
	return bins

def levels_from_file(filename):
	with open(filename) as f:
		lst = [float(line.rstrip()) for line in f]
	return lst

def get_level_maker(levels_file):
	levels_list = levels_from_file(levels_file)
	def get_level(value):
		level = 0
		for level_bin in levels_list:
			if value > level_bin:
				level +=1
			else:
				return level
		return level

	no_of_levels = len(levels_list) + 1
	return get_level, no_of_levels

#OBSOLETE
def convert_value_to_level_sequence(value_sequence, bins):
	levels = []
	for value in value_sequence:
		level = 0
		for bin_no, bin_upper_limit in enumerate(bins):
			if value > bin_upper_limit:
				level += 1
			else:
				break
		levels.append(level)
	return levels

def reducePuncCode(puncCode):
	if puncCode in [4, 5, 6, 7, 8]: #period
		return 2
	else:
		return puncCode

def reducePunc(punc):
	if punc and not punc.isspace():
		puncCode = INV_PUNCTUATION_CODES[punc]
		reducedPuncCode = reducePuncCode(puncCode)
		return PUNCTUATION_VOCABULARY[reducedPuncCode]
	else:
		return punc
