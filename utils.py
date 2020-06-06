import torch
import torch.nn as nn

def cross_entropy_mse(pred_win_prob, neural_net_prob, self_play_winner, search_prob, parameters, c = 1.00):
	'''
	This function takes the pred_win_prob v and the neural network move probabilities p, and combines two loss functions: the MSE
	of the v with the true winner of the self-played game z, and the cross entropy loss of the move probabilities and the search probablities from the MCTS
	'''
	

	raise NotImplementedError

def train(train_data):
	'''
	train neural network optimizing on loss function created above

	'''

	raise NotImplementedError

def test(net, test_data):
	'''
	
	'''
	raise NotImplementedError

