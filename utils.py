import torch
import torch.nn as nn
import pytorch_lightning as pl 

class LitModel(pl.LightningModel):
	def __init__(self, board_size: int, residual_layers : int):
		super().__init__
		'''
		class takes the game board size and the amount of residual layers
		'''

	def forward(self, x):

		raise NotImplementedError

	def prepare_data(self):
		'''
		prepare game data/MCTS data
		can probably do the random validation set data picking here
		'''
		raise NotImplementedError


	def train_dataloader(self):
		'''
		load test data
		'''
		raise NotImplementedError



	def configure_optimizer(self):
		'''
		momentum set according to paper, learning rate can be tweeked

		'''
		optimizer = torch.optim.SGD(self.parameters, lr = 10e-2, momentum = 0.9, weight_decay = 1.0)
		return optimizer

	def training_step(self, batch, batch_idx):
		'''
		training step
		'''
		raise NotImplementedError

	def validation_dataloader(self):
		'''
		load val data
		'''
		raise NotImplementedError

	def validation_step(self, batch, batch_idx):
		'''
		validation step
		'''
		raise NotImplementedError

	def validation_epoch_end(self, outputs):
		'''
		prints validation error at end of epoch
		'''
		raise NotImplementedError

	def test_dataloader(self):
		'''
		load testdata
		'''
		raise NotImplementedError

	def test_step(self, batch, batch_idx):
		'''
		testing step
		'''
		raise NotImplementedError

	def test_epoch_end(self, outputs):
		'''
		prints test error at end of epoch
		'''
		raise NotImplementedError



def cross_entropy_mse(pred_win_prob, neural_net_prob, self_play_winner, search_prob, parameters):
	'''
	This function takes the pred_win_prob v and the neural network move probabilities p, and combines two loss functions: the MSE
	of the v with the true winner of the self-played game z, and the cross entropy loss of the move probabilities and the search probablities from the MCTS. 
	I think this can probably go into the Lighting Module though.

	'''

	raise NotImplementedError
