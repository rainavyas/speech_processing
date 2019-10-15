import torch
from trained_models import ThreeLayerNet_1LevelAttn_RNN_multihead



# Define adversarial attack class
class attack_part3(torch.nn.Module):
	def __init__(self, word_num, init_word_vects):
	
		super(attack_part3, self).__init__()
		
		# Initialise adversarial vectors
		self.vects = torch.nn.ParameterList([torch.nn.Parameter(init_word_vects[i]) for i in range(word_num)])
		

		# Load trained model
		model_path = '../saved_models/part3_RNN_multihead_seed1.pt'
		model = torch.load(model_path)
		model.eval()
		self.model = model

	
	def forward(self, X, M, L, Qs):
		
		'''
		Expect M and L to already be correctly modified
		for extra number of words
		Expect Qs to be a list of tensors Q
		Q_i is for adding adversarial word i
		Q_i is a 3D tensor bs x utt_num x max_words
		with zeros everywhere apart from in L+i positions
		where there is a 1 
		NOTE: Q comes in as a tensor with first dimension batch, second dim the Qs index
		'''

		# Loop through adversarial words to add
		for i, vect in enumerate(self.vects):
			# Represent vect as 4D tensor 1x1x1x300
			expanded_vect = vect[None, None, None, :]
			
			# Extract ith slice of Qs
			Qi = Qs[:, i, :, :]
			
			# Expand Qs_ith slice to be 4D tensor with added 4th dimension
			expanded_Q = Qi[:, :, :, None]
			
			# Exploit broadcasting in multiplication
			P = expanded_Q * expanded_vect

			# Append the adversarial word vector to input data
			X = X.add(P)

		
		# Pass through the trained model
		y_pred = self.model(X, M, L)

		return y_pred
