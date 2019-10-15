import json
import torch
from models import ThreeLayerNet_1LevelAttn_multihead


#Define previous sub-optimal words found
words = []

word_num= len(words)+1
max_words_in_utt = 200


#Select any model that accepts extra word vectors to append
model_path = 'part3_multihead_seed1.pt'
model = torch.load(model_path)
model.eval()

#Preprocess evaluatiion data
#Load evaluation data
input_data = 'eval_data4D_training_part3.txt'

with open(input_data, 'r') as f:
        data = json.load(f)

print("Loaded data")

# Extract relevant parts of data
X = data[0]
y = data[1]
L_list = data[2]
y_overall = data[3]

# Convert to tensors
X = torch.FloatTensor(X)
y = torch.FloatTensor(y)
L = torch.FloatTensor(L_list)
y_overall = torch.FloatTensor(y_overall)

# Remove 0s for RNN based model
L[L==0]=1

Qs = []

for i in range(word_num):
        Q = []
        for row_item in L_list:
                Q_row = []
                for col_item in row_item:
                        new_vect = [0]*max_words_in_utt
                        new_vect[col_item+i] = 1
                        Q_row.append(new_vect)
                Q.append(Q_row)
        Qs.append(Q)

Qs = torch.FloatTensor(Qs)
# Make the first dimension batch and second dimension iterate through the Qs
Qs = torch.transpose(Qs, 0, 1)


# Add word_num to every length to account for adversarial words
L = torch.add(L, word_num)

# Make the mask from utterance lengths matrix L
M = [[([1]*utt_len + [-100000]*(X.size(2)- utt_len)) for utt_len in speaker] for speaker in L_list]
M = torch.FloatTensor(M)






#Load word2vec dict
file_to_read = 'word2vec_emb.txt'
with open(file_to_read, 'r') as f:
	word2vec_dict = json.loads(f.read())


#Define sub-optimal words already learnt greedily
vects = [[float(num) for num in word2vec_dict[word]] for word in words]

# Load ASR and word2vec overlap words that we want to check through
words_file = 'test_words.txt'
with open(words_file, 'r') as f:
	test_words = json.loads(f.read())




# Define class to have a structure in place to keep num_words best adversarial words
class best_words:
	def __init__(self, num_words):
		self.words = [['none', 0]]*num_words
	
	def check_word_to_be_added(self, y_avg):
		if y_avg > self.words[-1][1]:
			return True
		else:
			return False

	def add_word(self, word, y_avg):
		self.words.append([word, y_avg])
		# Sort from highest to lowest y_avg
		self.words = sorted(self.words, reverse = True, key = lambda x: x[1])
		# Drop the worst extra word
		self.words = self.words[:-1]






# Exhaustively search test_words words for next word to add
best = best_words(5)

for word in test_words:
	
	word_vect = word2vec_dict[word]
	new_vect = [float(num) for num in word_vect]
	init_word_vects = vects[:]
	init_word_vects.append(new_vect)
	init_word_vects = [torch.FloatTensor(vect) for vect in init_word_vects]

	Xp = X
	for i, vect in enumerate(init_word_vects):
		# Represent vect as 4D tensor 1x1x1x300
		expanded_vect = vect[None, None, None, :]
		
		# Extract ith slice of Qs
		Qi = Qs[:, i, :, :]

		#Expand Qs_ith slice to be 4D tensor with added 4th dimension
		expanded_Q = Qi[:, :, :, None]

		# Exploit broadcasting in multiplication
		P = expanded_Q * expanded_vect

		# Append the adversarial word vector to input data
		Xp = Xp.add(P)

                # Pass through the trained model
	y_adv = model(Xp, M)
	
	
	y_adv = y_adv[:,0]
	y_adv[y_adv>6] = 6
	y_adv[y_adv<0] = 0

	val = torch.mean(y_adv).item()
	
	if best.check_word_to_be_added(val):
		best.add_word(word, val)
		print(best.words)
		

print(best)







	
	
