import json
import torch
from transformers import *
from models import ThreeLayerNet_1LevelAttn_RNN
import matplotlib.pyplot as plt
import numpy as np

# adversarial words to add:
adv_words = ['', 'detainees', 'awfully', 'rabies', 'admittedly', 'foreclosures', 'admittedly']



# Define constants
MAX_UTTS_PER_SPEAKER_PART = 1
MAX_WORDS_IN_UTT = 200






# Make sentences from adv words
sents = []
for i in range(len(adv_words)):
	new_sen = ''
	for word in adv_words[:i+1]:
		new_sen = new_sen + ' ' + word

	sents.append(new_sen)


part = 3
seed = 1
trained_model_to_load = 'part'+str(part)+'_multihead_seed'+str(seed)+'.pt'

#Load the model to attack
model_path = '../saved_bert_models/'+trained_model_to_load
model = torch.load(model_path)
model.eval()


# Load the relevant part 'useful' data
data_file = '../data/BLXXXeval3/useful_part'+str(part)+'.txt'
with open(data_file, 'r') as f:
	utterances = json.loads(f.read())

print("Loaded word Data")

# Convert json output from unicode to string
utterances = [[str(item[0]), str(item[1])] for item in utterances]





#Load tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_basic_tokenize=False, do_lower_case=True)
bert_model = BertModel.from_pretrained('bert-base-cased')
bert_model.eval()

print("Loaded BERT model")



# dict to store the part3 true expert grade
grades = {}

# Get the expert grades by utterance id
grades_file_path = '/home/alta/BLTSpeaking/grd-graphemic-kmk-v2/GKTS4-D3/grader/BLXXXeval3/score_10_mbr_rnnlm/data/grades.txt'
lines = [str(line.rstrip('\n')) for line in open(grades_file_path)]


for line in lines[1:]:
	speaker_id = line[:12]
	grade_part_3 = line[-15:-12]
	grades[speaker_id] = float(grade_part_3)


y_preds = []
y_reals = [] # as dict used y_real in different order every time we do bert process
# For each length adversarial phrase find prediction and add to the same graph
for sent in sents:
	
	# Add the advserarial phrase to every utterance
	attacked_utts = [[item[0], item[1]+' '+sent] for item in utterances]


	# Convert sentences to a list of BERT embeddings
	# Store as dict of speaker id to utts list (list of embeddings)
	utt_embs = {}	
	for item in attacked_utts:
		fileName = item[0]
		speakerid = fileName[:12]
		sentence = item[1]
		
		tokenized_text = tokenizer.tokenize(sentence)
		indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
		if len(indexed_tokens) < 1:
			word_vecs = [[0]*768]
		else:
			tokens_tensor = torch.tensor([indexed_tokens])
			with torch.no_grad():
				encoded_layers, _ = bert_model(tokens_tensor)
			bert_embs = encoded_layers.squeeze(0)
			word_vecs = bert_embs.tolist()
		if speakerid not in utt_embs:
			utt_embs[speakerid] = [word_vecs]
		else:
			utt_embs[speakerid].append(word_vecs)
		
		


	# Convert to appropriate 4D tensor
	y_real = []
	vals = []
	for speaker_id in utt_embs:
		try:
			y_real.append(grades[speaker_id])
		except:
			continue
		vals.append(utt_embs[speaker_id])
	y_reals.append(y_real)
	
	# Initialise list to hold all input data in tensor format
	X = []

	# Initialise 2D matrix format to store all utterance lengths per speaker
	utt_lengths_matrix = []

	for utts in vals:
		new_utts = []
		
		# Reject speakers with not exactly correct number of utterances in part
		if len(utts) != MAX_UTTS_PER_SPEAKER_PART:
			continue

		# Create list to store utterance lengths
		utt_lengths = []

		for curr_utt in utts:
			num_words = len(curr_utt)

			if num_words <= MAX_WORDS_IN_UTT:
				# append padding of zero vectors
				words_to_add = MAX_WORDS_IN_UTT - num_words
				zero_vec_word = [0]*768
				zero_vec_words = [zero_vec_word]*words_to_add
				new_utt = curr_utt + zero_vec_words
				utt_lengths.append(num_words)
			else:
				# Shorten utterance from end
				new_utt = curr_utt[:MAX_WORDS_IN_UTT]
				utt_lengths.append(MAX_WORDS_IN_UTT)

			# Convert all values to float
			new_utt = [[float(i) for i in word] for word in new_utt]

			new_utts.append(new_utt)

		X.append(new_utts)
		utt_lengths_matrix.append(utt_lengths)

	# Convert to tensors
	X = torch.FloatTensor(X)
	L = torch.FloatTensor(utt_lengths_matrix)

	# Make the mask from utterance lengths matrix L
	M = [[([1]*utt_len + [-100000]*(X.size(2)- utt_len)) for utt_len in speaker] for speaker in utt_lengths_matrix]
	M = torch.FloatTensor(M)


	# Pass through the trained model to get y_predictions
	y_adv = model(X, M)

	y_adv = y_adv[:, 0]
	y_adv[y_adv>6] = 6
	y_adv[y_adv<0] = 0

	y_preds.append(y_adv)



# Plot graphs of pred, adv vs real
y_real0 = y_reals[0]
y_preds = [y_pred.tolist() for y_pred in y_preds]

y_pred = y_preds[0]
y_advN = y_preds[1:]


for i in range(len(y_advN)):
	y_adv = y_advN[i]
	plt.plot(y_real0, y_pred, 'o', color = 'black', label = 'Speaker data point')
	plt.plot(y_reals[i+1], y_adv, 'o', color = 'green', label = 'Adversarially attacked data point')
	plt.plot(y_reals[i+1], y_reals[i+1], color='red', label='Optimal Prediction')
	plt.plot(np.unique(y_real0), np.poly1d(np.polyfit(y_real0, y_pred, 1))(np.unique(y_real0)), label = 'No attack LOB, k=0')
	plt.plot(np.unique(y_reals[i+1]), np.poly1d(np.polyfit(y_reals[i+1], y_adv, 1))(np.unique(y_reals[i+1])), label = 'Adversarial LOB, k='+str(i+1))


	plt.xlabel('True Part ' + str(3)+ ' Expert Grades')
	plt.ylabel('Predicted Grades')
	plt.legend(loc='lower right')

	png_file = 'plots/adv_part3_k'+str(i+1)
	plt.savefig(png_file)
	plt.clf()






 
 
