import torch
import argparse


# This is just a test file for trying different things


# Get the part and its seed to adversarially attack
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('--part', type = int, default = 3, help = 'Specify the part to attack')
commandLineParser.add_argument('--seed', type = int, default = 1, help = 'Specify the seed of particular trained part model')

args = commandLineParser.parse_args()
part = args.part
seed = args.seed

print("part: " + str(part))
print("seed: " + str(seed))

