
'''
Use evolutionary method to get UNIVERSAL adversarial attack (repeating noise segment to add to any wav file to boost output score)

1) Load wav files of bulats data (rebranded as LINGUAskills General)

2) Initialise N noise vectors, each 8 frames in size

3) Superimpose each noise vector to all input wav files (i.e. have N sets of altered bulats inputs)

4) Generate plp vector files for altered wav files

5) Also generate plp vector files without added noise, call this plp* -- these plp files should already exist

6) Pass ASR output from plp* and the plp generated files through deep_pron system

7) Average output predicted grade

8) Use this to evaluate fitness score: a) higher grade + similarity of plp files (averaged across all data input) gives fitness score

'''

# NOTE: Current evolutionary implementation does not invlove mutation of the children 

import os
from random import random
from random import seed
from scipy.io import wavfile
import scandir
import subprocess
import numpy as np




def generate_noise_unit():
	'''
	Make 8 frame random noise unit
	'''

	NUM_FRAMES = 8
	MAX_AMP = 1000

	noise_vals = [0]*NUM_FRAMES
	noise_vals = [MAX_AMP*random() - MAX_AMP/2 for i in noise_vals]
	
	return noise_vals




def add_periodic_noise(signal, noise_unit):
	num_repeats_required = signal.shape[0]/len(noise_unit)
	extra_vals_required = signal.shape[0] - (num_repeats_required*len(noise_unit))

	extra_vals = noise_unit[:extra_vals_required]

	periodic_noise = []
	for count in range(num_repeats_required):
		periodic_noise += noise_unit

	periodic_noise += extra_vals
	periodic_noise = np.array(periodic_noise)

        
	# Add the noise to the signal
	combined = np.add(signal, periodic_noise)

	return combined







def make_modified_wav_files(noise_unit):
	'''
	Add the noise unit periodically to each wav file in the source directory
	and write the output to the destination directory
	'''

	SOURCE_WAV_FOLDER = '/home/alta/BLTSpeaking/grd-graphemic-vr313/speech_processing/merger/adversarial/evolutionary/data/BLXXXgrd02_original_wav'
	DESTINATION_WAV_FOLDER = '/home/alta/BLTSpeaking/grd-graphemic-vr313/speech_processing/merger/adversarial/evolutionary/data/BLXXXgrd02_altered_wav'

	# get list of subfolders in the source wav folder
	wav_sub_folders = [f.name for f in scandir.scandir(SOURCE_WAV_FOLDER) if f.is_dir()]

	# To save time, only keep 10 sub-folders
	wav_sub_folders = [:10]

	for sub_folder in wav_sub_folders:
		
		wav_files = [f.name for f in scandir.scandir(SOURCE_WAV_FOLDER+"/"+sub_folder)]

		for wav_file in wav_files:

			rate, data = wavfile.read(SOURCE_WAV_FOLDER+"/"+sub_folder+"/"+wav_file)
			modified_data = add_periodic_noise(data, noise_unit)
			
			# Write the modified data to a destination wav file
			wavfile.write(DESTINATION_WAV_FOLDER+"/"+sub_folder+"/"+wav_file, rate, modified_data)

	

def generate_plp_vectors():
	'''
	Generate plp vector file per altered wav file
	using the existing coding file to define the mapping from wav to plp
	'''
	
	# Call bash script to delete all current plp files and generate new plp files
	subprocess.call("./make_plp.sh")

	# Now wait till all plp files generated in subfolder
	NUM_FILES = 20863
	destination_folder = '/home/alta/BLTSpeaking/grd-graphemic-vr313/speech_processing/merger/adversarial/evolutionary/data/BLXXXgrd02_altered_plp'
	num_files = len(os.walk(destination_folder).next()[2])

	while num_files < NUM_FILES:
		num_files = len(os.walk(destination_folder).next()[2])


	# One extra pause
	pass
	


def get_similarity_score():
	'''
	Compare plp vector files in original to altered
	compute average mse difference
	return reciprocal (so higher score better)
	Scale sim_score to be of order of magnitude of avg_grade
	'''

	'''original_folder = '/home/alta/BLTSpeaking/grd-graphemic-vr313/speech_processing/merger/adversarial/evolutionary/data/BLXXXgrd02_original_plp'
	altered_folder = '/home/alta/BLTSpeaking/grd-graphemic-vr313/speech_processing/merger/adversarial/evolutionary/data/BLXXXgrd02_altered_plp'

	plp_files = [f.name for f in scandir.scandir(original_folder)]

	# To save time, perform average over only 100 plp files
	plp_files = plp_files[:100]

	# get mse differences for each file
	for file_name in plp_files:
	'''

	# For now, ignore similarity score and return 1 for everything
	return 1.0	



def calc_score(noise_unit):
	'''
	Compute the average predicted grade and the similarity score (using plp files)
	'''

	# Add the noise vector periodically to each raw wav file
	make_modified_wav_files(noise_unit)

	# Now generate the plp vector files using existing coding file
	generate_plp_vectors()
	
	# Compute the similarity score from the plp vector files
	sim_score = get_similarity_score()

	# Generate the pickle file from existing SCP file
	subprocess.call('./make_altered_pkl.sh')	

	# Generate grades file from the pickle file generated
	subprocess.call('./generate_grades.sh')

	# Compute the average predicted grade
	





def compute_scores(population, last_half):
	'''
	Compute the similarity score, avg grade pred and then overall score
	for each member of the population
	If last_half is TRUE then scores only need to be computed for the second half of the population
	as first half scores have already been calculated
	'''
	if last_half:
		start = 10
		population_with_scores = population[:10]
	else:
		start = 0
		population_with_scores = []
	
	for member in population:
		noise_unit = member[0]
		grade_pred, sim_score = calc_scores(noise_unit)
		overall_score = compute_overall_score(grade_pred, sim_score)
		
		item = (noise_unit, grade_pred, sim_score, overall_score)
		population_with_scores.append(item) 

	return population_with_scores


'''
# Seed for reproducibility
seed = 1


# Define size of population
N = 20


# Create list, where each element is a tuple: (noise unit vector, av predicted grade, similarity score, overall score)
population = []


# Initialise the population 
for member in range(N):
	grade_pred = -1
	sim_score = -1
	overall_score = -1
	new_noise_unit = generate_noise_unit()
	
	item = (new_noise_unit, grade_pred, sim_score, overall_score)
	population.append(item)

population = compute_scores(population, last_half = False)



# Perform evolutionary iteration
NUM_ITERS = 5

for iter in range(NUM_ITERS):

	# Write the current members (noises) to a file for record


	# Compute the scores for each member of the population, only the last half are required as first 10 are parents, second 10 children
	population = compute_scores(population, last_half = True)
	
	# Make new population by taking 10 best parents and 10 mutated children from the 10 best parents
	# Generate 5 pairs from 10 best parents
	# Each pair gives 2 children (2/3 similar to each parent)
	
	# Sort list of members (population) by the score
	ranked_population = sorted(population, key = lambda x: x[3], reverse = True)
		
	# Keep 10 best parents
	best_parents = ranked_population[:10]

	# Make 5 pairs deterministically: 1+2, 3+4, ... , 9+10
	# Compute the children per pair

	children = []
	for index in range(0, len(population), 2):
		parent1 = np.array(population[index][0])
		parent2 = np.array(population[index+1][0])
		
		child1 = list(((2/3)*parent1) + ((1/3)*parent2))
		child2 = list(((1/3)*parent1) + ((2/3)*parent2))

		item1 = (child1, -1, -1, -1)
		item2 = (child2, -1, -1, -1)
		
		children.append(item1)
		children.append(item2)

	# Make the new population
	population = ranked_population + children
'''		
