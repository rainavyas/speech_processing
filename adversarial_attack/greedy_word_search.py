import argparse
import json
import torch
from model_structures import ThreeLayer_1LevelAttn_RNN


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

 
