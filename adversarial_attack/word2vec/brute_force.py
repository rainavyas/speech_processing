import json
import torch
from models import attack_part3


#Define previous sub-optimal words found
words = ['www.zawya.com', 'spokesman_micky_rosenfeld', 'average_+_-1', 'viewpoints_race']

word_num= len(words)+1
max_words_in_utt = 200


#Select any model that accepts extra word vectors to append
model_path = '../saved_models/part3_RNN_multihead_seed1.pt'
model = torch.load(model_path)
model.eval()

#Preprocess evaluatiion data
#Load evaluation data
target_file3 = '../eval_data4D_training_part3.txt'

with open(target_file3, 'r') as f:
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
file_to_read = '../../word2vec_emb.txt'
with open(file_to_read, 'r') as f:
	word2vec_dict = json.loads(f.read())


#Define sub-optimal words already learnt greedily
vects = [[float(num) for num in word2vec_dict[word]] for word in words]



# Exhaustively search word2vec words for next word to add
best = ['none', 0]
for word, word_vect in word2vec_dict.items():
	
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
	y_adv = model(Xp, M, L)
	
	
	y_adv = y_adv[:,0]
	y_adv[y_adv>6] = 6
	y_adv[y_adv<0] = 0

	val = torch.mean(y_adv)
	
	if val > best[1]:
		best = [word, val]
		print(words, best)

print(best)







	
	
