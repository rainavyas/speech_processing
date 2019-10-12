import argparse
import json
import torch
from models import ThreeLayerNet_1LevelAttn_RNN


# Get the part and its seed to adversarially attack
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('--part', type = int, default = 1, help = 'Specify the part to attack')
commandLineParser.add_argument('--seed', type = int, default = 1, help = 'Specify the seed of particular trained part model')

args = commandLineParser.parse_args()
part = args.part
seed = args.seed

trained_model_to_load = 'part'+str(part)+'_multihead_seed'+str(seed)+'.pt'

#Load the model to attack
model_path = '../saved_bert_models/'+trained_model_to_load
model = torch.load(model_path)
model.eval()


# Load the relevant part 'useful' data
data_file = '../data/BLXXXeval3/useful_part'+str(part)+'.txt'
with open(data_file, 'r') as f:
	utterances = json.loads(f.read())

# Convert json output from unicode to string
utterances = [[str(item[0]), str(item[1])] for item in utterances]


# Create list of words to iterate through for appending
words_file = '/home/alta/BLTSpeaking/lms/LM15-grph/wlists/train.lst'
with open(words_file, 'r') as f:
	test_words = f.readlines()
test_words = [str(word).lower() for word in test_words]
print(len(test_words))

for i in range(20):
	print(test_words[i])
 
