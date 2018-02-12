# coding: utf-8
from __future__ import division

import theano
from theano import tensor as T, function, printing
import _pickle
import os
import numpy as np

def PReLU(a, x):
    return T.maximum(0.0, x) + a * T.minimum(0.0, x)

def ReLU(x):
    return T.maximum(0.0, x)

def _get_shape(i, o, keepdims):
    if (i == 1 or o == 1) and not keepdims:
        return (max(i,o),)
    else:
        return (i, o)

def _slice(tensor, size, i):
    """Gets slice of columns of the tensor"""
    if tensor.ndim == 2:
        return tensor[:, i*size:(i+1)*size]
    elif tensor.ndim == 1:
        return tensor[i*size:(i+1)*size]
    else:
        raise NotImplementedError("Tensor should be 1 or 2 dimensional")

def weights_const(i, o, name, const, keepdims=False):
    W_values = np.ones(_get_shape(i, o, keepdims)).astype(theano.config.floatX) * const
    return theano.shared(value=W_values, name=name, borrow=True)

def weights_identity(i, o, name, const, keepdims=False):
    #"A Simple Way to Initialize Recurrent Networks of Rectified Linear Units" (2015) (http://arxiv.org/abs/1504.00941)
    W_values = np.eye(*_get_shape(i, o, keepdims)).astype(theano.config.floatX) * const
    return theano.shared(value=W_values, name=name, borrow=True)

def weights_Glorot(i, o, name, rng, is_logistic_sigmoid=False, keepdims=False):
    #i: no of levels
    #o: output vector size
    #http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    d = np.sqrt(6. / (i + o))
    if is_logistic_sigmoid:
        d *= 4.
    W_values = rng.uniform(low=-d, high=d, size=_get_shape(i, o, keepdims)).astype(theano.config.floatX)
    return theano.shared(value=W_values, name=name, borrow=True)

class GRULayer(object):

    def __init__(self, rng, n_in, n_out, minibatch_size):
        super(GRULayer, self).__init__()
        # Notation from: An Empirical Exploration of Recurrent Network Architectures

        self.n_in = n_in
        self.n_out = n_out

        # Initial hidden state
        self.h0 = theano.shared(value=np.zeros((minibatch_size, n_out)).astype(theano.config.floatX), name='h0', borrow=True)

        # Gate parameters:
        self.W_x = weights_Glorot(n_in, n_out*2, 'W_x', rng, keepdims=True)
        self.W_h = weights_Glorot(n_out, n_out*2, 'W_h', rng, keepdims=True)
        self.b = weights_const(1, n_out*2, 'b', 0)
        # Input parameters
        self.W_x_h = weights_Glorot(n_in, n_out, 'W_x_h', rng, keepdims=True)
        self.W_h_h = weights_Glorot(n_out, n_out, 'W_h_h', rng, keepdims=True)
        self.b_h = weights_const(1, n_out, 'b_h', 0)

        self.params = [self.W_x, self.W_h, self.b, self.W_x_h, self.W_h_h, self.b_h]

    def step(self, x_t, h_tm1):

        rz = T.nnet.sigmoid(T.dot(x_t, self.W_x) + T.dot(h_tm1, self.W_h) + self.b)
        r = _slice(rz, self.n_out, 0)
        z = _slice(rz, self.n_out, 1)

        h = T.tanh(T.dot(x_t, self.W_x_h) + T.dot(h_tm1 * r, self.W_h_h) + self.b_h)

        h_t = z * h_tm1 + (1. - z) * h

        return h_t

def load_stage2(file_path, minibatch_size, stage1_model_file_name):
	import models
	import _pickle
	import theano
	import numpy as np

	with open(file_path, 'rb') as f:
		state = _pickle.load(f)

	Model = getattr(models, state["type"])
	print(Model)

	rng = np.random
	rng.set_state(state["random_state"])

	stage1_net, stage1_inputs, stage1_input_feature_names, _ = load(stage1_model_file_name, minibatch_size)
	x_tensor = stage1_inputs[0]

	for tensor_info in state["input_tensors_info"]:
		if tensor_info['name'] == "word":
			#x_tensor = T.imatrix(tensor_info['name'])
			x_PuncTensor_num_hidden = tensor_info['size_hidden']
			x_PuncTensor_size_emb=tensor_info['size_emb']
			x_vocabulary_size = tensor_info['vocabulary_size']
		elif tensor_info['name'] == "pause_before":
			p_tensor = T.matrix(tensor_info['name'])
			p_PuncTensor_num_hidden = tensor_info['size_hidden']

	x_PuncTensor = PuncTensor(name="word", tensor=x_tensor, size_hidden=x_PuncTensor_num_hidden, size_emb=x_PuncTensor_size_emb, vocabularized=True, vocabulary_size=x_vocabulary_size, bidirectional=True)
	p_PuncTensor = PuncTensor(name="pause_before", tensor=p_tensor, size_hidden=p_PuncTensor_num_hidden, size_emb=1, vocabularized=False, bidirectional=False)

	input_PuncTensors = [x_PuncTensor, p_PuncTensor]
	input_feature_names = ["word", "pause_before"]

	net = Model(rng=rng,
				y_vocabulary_size=state["y_vocabulary_size"],
				minibatch_size=minibatch_size,
				num_hidden_output=state["num_hidden_output"],
				x_PuncTensor=x_PuncTensor,
				p_PuncTensor=p_PuncTensor,
				stage1_net=stage1_net,
				stage1_inputs=stage1_inputs,
				stage1_input_feature_names=stage1_input_feature_names)

	for net_param, state_param in zip(net.params, state["params"]):
		net_param.set_value(state_param, borrow=True)

	gsums = [theano.shared(gsum) for gsum in state["gsums"]] if state["gsums"] else None

	tensors = [i.tensor for i in input_PuncTensors]

	return net, tensors, input_feature_names, (gsums, state["learning_rate"], state["validation_ppl_history"], state["epoch"], rng)

def load(file_path, minibatch_size, first_stage_file=None):
	import models
	import _pickle
	import theano
	import numpy as np

	with open(file_path, 'rb') as f:
		state = _pickle.load(f)

	Model = getattr(models, state["type"])
	print(Model)

	rng = np.random
	rng.set_state(state["random_state"])

	input_PuncTensors = []
	input_feature_names = []

	for tensor_info in state["input_tensors_info"]:
		input_feature_names.append(tensor_info['name'])
		if tensor_info['bidirectional']:
			is_bidi = True
		else:
			is_bidi = False
		if tensor_info['vocabularized']:
			tensor = T.imatrix(tensor_info['name'])
			vocabulary_size = tensor_info['vocabulary_size']
			feature_PuncTensor = PuncTensor(name=tensor_info['name'], tensor=tensor, size_hidden=tensor_info['size_hidden'], size_emb=tensor_info['size_emb'], vocabularized=True, vocabulary_size=tensor_info['vocabulary_size'], bidirectional=is_bidi)
			print("loaded %s"%feature_PuncTensor.name)
		else:
			tensor = T.matrix(tensor_info['name'])
			feature_PuncTensor = PuncTensor(name=tensor_info['name'], tensor=tensor, size_hidden=tensor_info['size_hidden'], size_emb=tensor_info['size_emb'], vocabularized=False, vocabulary_size=tensor_info['vocabulary_size'], bidirectional=is_bidi)
			print("loaded %s"%feature_PuncTensor.name)
		input_PuncTensors.append(feature_PuncTensor)

	net = Model(rng=rng,
        minibatch_size=minibatch_size,
        y_vocabulary_size=state["y_vocabulary_size"],
        num_hidden_output=state["num_hidden_output"],
        input_tensors=input_PuncTensors
	)

	for net_param, state_param in zip(net.params, state["params"]):
		net_param.set_value(state_param, borrow=True)

	gsums = [theano.shared(gsum) for gsum in state["gsums"]] if state["gsums"] else None

	tensors = [i.tensor for i in input_PuncTensors]

	return net, tensors, input_feature_names, (gsums, state["learning_rate"], state["validation_ppl_history"], state["epoch"], rng)

class PuncTensor(object):
	def __init__(self, name, tensor=None, size_hidden=0, size_emb=1, vocabularized=False, vocabulary_size = 0,  bidirectional=False):
		#if vocabularized == False, size_emb has to be 1
		self.name = name
		self.tensor = tensor
		self.bidirectional = bidirectional
		self.vocabularized = vocabularized
		self.vocabulary_size = vocabulary_size
		self.size_hidden = size_hidden
		self.size_emb = size_emb

		self.GRU_forward = None
		self.GRU_backward = None

		self.We = None

	def initialize_layers(self, rng, minibatch_size):
		if self.vocabularized:
			self.We = weights_Glorot(self.vocabulary_size, self.size_emb, 'We_' + self.name, rng)

		total_n_out = 0
		self.GRU_forward = GRULayer(rng=rng, n_in=self.size_emb, n_out=self.size_hidden, minibatch_size=minibatch_size)
		if self.bidirectional:
			self.GRU_backward = GRULayer(rng=rng, n_in=self.size_emb, n_out=self.size_hidden, minibatch_size=minibatch_size)
			total_n_out += self.size_hidden * 2
		else:
			total_n_out += self.size_hidden

		return total_n_out

	def is_empty(self):
		if self.tensor == None:
			return True
		else:
			return False

	def as_dict(self):
		tensor_info = { 'name':self.name,
						'size_hidden':self.size_hidden,
						'size_emb':self.size_emb,
						'vocabularized':self.vocabularized,
						'vocabulary_size':self.vocabulary_size,
						'bidirectional':self.bidirectional}
		return tensor_info

class GRU_parallel(object):
	def __init__(self, rng, y_vocabulary_size, minibatch_size, num_hidden_output, input_tensors):
		self.used_input_tensors = [tensor for tensor in input_tensors if not tensor.is_empty()]
		self.input_feature_names = [tensor.name for tensor in self.used_input_tensors]
		self.vocabularized_feature_names = [tensor.name for tensor in self.used_input_tensors if tensor.vocabularized]
		self.num_hidden_output = num_hidden_output
		self.y_vocabulary_size = y_vocabulary_size

		n_attention = 0
		for tensor in self.used_input_tensors:
			n_attention += tensor.initialize_layers(rng, minibatch_size)

		print("concatenated layers size: %i"%n_attention)

		# output model
		self.GRU = GRULayer(rng=rng, n_in=n_attention, n_out=num_hidden_output, minibatch_size=minibatch_size)  #DIKKAT
		self.Wy = weights_const(num_hidden_output, self.y_vocabulary_size, 'Wy', 0)
		self.by = weights_const(1, self.y_vocabulary_size, 'by', 0)

		# attention model
		self.Wa_h = weights_Glorot(num_hidden_output, n_attention, 'Wa_h', rng) # output model previous hidden state to attention model weights
		self.Wa_c = weights_Glorot(n_attention, n_attention, 'Wa_c', rng) # contexts to attention model weights
		self.ba = weights_const(1, n_attention, 'ba', 0)
		self.Wa_y = weights_Glorot(n_attention, 1, 'Wa_y', rng) # gives weights to contexts

		# Late fusion parameters
		self.Wf_h = weights_const(num_hidden_output, num_hidden_output, 'Wf_h', 0)
		self.Wf_c = weights_const(n_attention, num_hidden_output, 'Wf_c', 0)
		self.Wf_f = weights_const(num_hidden_output, num_hidden_output, 'Wf_f', 0)
		self.bf = weights_const(1, num_hidden_output, 'bf', 0)

		self.params = [tensor.We for tensor in self.used_input_tensors if tensor.vocabularized]
		self.params += [self.Wy, self.by,
                        self.Wa_h, self.Wa_c, self.ba, self.Wa_y,
                        self.Wf_h, self.Wf_c, self.Wf_f, self.bf]
		self.params += self.GRU.params
		for tensor in self.used_input_tensors:
			if tensor.bidirectional:
				self.params += tensor.GRU_forward.params
				self.params += tensor.GRU_backward.params
			else:
				self.params += tensor.GRU_forward.params
		# recurrence functions
		def output_recurrence(x_t, h_tm1, Wa_h, Wa_y, Wf_h, Wf_c, Wf_f, bf, Wy, by, context, projected_context):
			# Attention model
			h_a = T.tanh(projected_context + T.dot(h_tm1, Wa_h))
			alphas = T.exp(T.dot(h_a, Wa_y))
			alphas = alphas.reshape((alphas.shape[0], alphas.shape[1])) # drop 2-axis (sized 1)
			alphas = alphas / alphas.sum(axis=0, keepdims=True)
			weighted_context = (context * alphas[:,:,None]).sum(axis=0)

			h_t = self.GRU.step(x_t=x_t, h_tm1=h_tm1)

			# Late fusion
			lfc = T.dot(weighted_context, Wf_c) # late fused context
			fw = T.nnet.sigmoid(T.dot(lfc, Wf_f) + T.dot(h_t, Wf_h) + bf) # fusion weights
			hf_t = lfc * fw + h_t # weighted fused context + hidden state

			z = T.dot(hf_t, Wy) + by
			y_t = T.nnet.softmax(z)

			return [h_t, hf_t, y_t, alphas]

		def create_bidi(GRU_forward, GRU_backward):
			def bidirectional_recurrence(x_f_t, x_b_t, h_f_tm1, h_b_tm1):
				h_f_t = GRU_forward.step(x_t=x_f_t, h_tm1=h_f_tm1)
				h_b_t = GRU_backward.step(x_t=x_b_t, h_tm1=h_b_tm1)
				return [h_f_t, h_b_t]
			return bidirectional_recurrence

		def create_unidi(GRU_layer):
			def unidirectional_recurrence(p_t, h_p_tm1):
				h_p_t = GRU_layer.step(x_t=p_t, h_tm1=h_p_tm1)
				return h_p_t	#dikkat i changed this
			return unidirectional_recurrence

		concatenated_input_tensors = []

		for tensor in self.used_input_tensors:
			if tensor.vocabularized:
				x = tensor.We[tensor.tensor.flatten()].reshape((tensor.tensor.shape[0], minibatch_size, tensor.size_emb))  
			else:
				x = tensor.tensor.flatten().reshape((tensor.tensor.shape[0], minibatch_size, 1))

			if tensor.bidirectional:
				bidi_input_recurrence = create_bidi(tensor.GRU_forward, tensor.GRU_backward)

				[h_f_t, h_b_t], _ = theano.scan(fn=bidi_input_recurrence,
		            sequences=[x, x[::-1]], # forward and backward sequences
		            outputs_info=[tensor.GRU_forward.h0, tensor.GRU_backward.h0])

				concatenated_input_tensors += [h_f_t, h_b_t[::-1]]
			else:
				unidi_input_recurrence = create_unidi(tensor.GRU_forward)

				h_p_t, _ = theano.scan(fn=unidi_input_recurrence,
		        	sequences=[x],
		        	outputs_info=[tensor.GRU_forward.h0])

				concatenated_input_tensors += [h_p_t]

		context = T.concatenate(concatenated_input_tensors, axis=2) 
		projected_context = T.dot(context, self.Wa_c) + self.ba

		[_, self.last_hidden_states, self.y, self.alphas], _ = theano.scan(fn=output_recurrence,
		sequences=[context[1:]], # ignore the 1st word in context, because there's no punctuation before that
		non_sequences=[self.Wa_h, self.Wa_y, self.Wf_h, self.Wf_c, self.Wf_f, self.bf, self.Wy, self.by, context, projected_context],
		outputs_info=[self.GRU.h0, None, None, None])

		print("Number of parameters is %d" % sum(np.prod(p.shape.eval()) for p in self.params))

		self.L1 = sum(abs(p).sum() for p in self.params)
		self.L2_sqr = sum((p**2).sum() for p in self.params)

	def cost(self, y):
		num_outputs = self.y.shape[0]*self.y.shape[1] # time steps * number of parallel sequences in batch
		output = self.y.reshape((num_outputs, self.y.shape[2]))
		return -T.sum(T.log(output[T.arange(num_outputs), y.flatten()]))

	def save(self, file_path, gsums=None, learning_rate=None, validation_ppl_history=None, best_validation_ppl=None, epoch=None, random_state=None):
		import _pickle
		input_tensors_info = [tensor.as_dict() for tensor in self.used_input_tensors]
		state = {
		    "type":                     	self.__class__.__name__,
		    "num_hidden_output":			self.num_hidden_output,
		    "input_feature_names":			self.input_feature_names,
		    "vocabularized_feature_names": 	self.vocabularized_feature_names,
		    "input_tensors_info":			input_tensors_info,
		    "y_vocabulary_size":        	self.y_vocabulary_size,
		    "params":                   	[p.get_value(borrow=True) for p in self.params],
		    "gsums":                    	[s.get_value(borrow=True) for s in gsums] if gsums else None,
		    "learning_rate":            	learning_rate,
		    "validation_ppl_history":   	validation_ppl_history,
		    "epoch":                    	epoch,
		    "random_state":             	random_state
		}

		with open(file_path, 'wb') as f:
			_pickle.dump(state, f)

class GRU_stage2(GRU_parallel):
	def __init__(self, rng, y_vocabulary_size, minibatch_size, num_hidden_output, x_PuncTensor, p_PuncTensor, stage1_net, stage1_inputs, stage1_input_feature_names):
		self.used_input_tensors = [x_PuncTensor, p_PuncTensor]
		x = x_PuncTensor.tensor
		p = p_PuncTensor.tensor

		self.stage1_net = stage1_net

		self.vocabularized_feature_names = [x_PuncTensor.name]
		self.y_vocabulary_size = y_vocabulary_size
		self.input_feature_names = [x_PuncTensor.name, p_PuncTensor.name]
		self.num_hidden_output = num_hidden_output

		# output model
		self.GRU = GRULayer(rng=rng, n_in=self.stage1_net.num_hidden_output + 1, n_out=num_hidden_output, minibatch_size=minibatch_size)
		self.Wy = weights_const(num_hidden_output, y_vocabulary_size, 'Wy', 0)
		self.by = weights_const(1, y_vocabulary_size, 'by', 0)

		self.params = [self.Wy, self.by]
		self.params += self.GRU.params

		def recurrence(x_t, p_t, h_tm1, Wy, by):

			h_t = self.GRU.step(x_t=T.concatenate((x_t, p_t.dimshuffle((0, 'x'))), axis=1), h_tm1=h_tm1)

			z = T.dot(h_t, Wy) + by
			y_t = T.nnet.softmax(z)

			return [h_t, y_t]

		[_, self.y], _ = theano.scan(fn=recurrence,
									 sequences=[self.stage1_net.last_hidden_states, p],
									 non_sequences=[self.Wy, self.by],
									 outputs_info=[self.GRU.h0, None])

		print("Number of parameters is %d" % sum(np.prod(p.shape.eval()) for p in self.params))
		print("Number of parameters with stage1 params is %d" % sum(np.prod(p.shape.eval()) for p in self.params + self.stage1_net.params))

		self.L1 = sum(abs(p).sum() for p in self.params)
		self.L2_sqr = sum((p**2).sum() for p in self.params)
