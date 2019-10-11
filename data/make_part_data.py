import json

input_file = 'BLXXXeval3/useful_data.txt'
target_file_prefix = 'BLXXXeval3/useful_part'

# Read in python list of utterances and their words
with open(input_file, 'r') as f:
	utterances = json.loads(f.read())

print("Collected useful data")

# Convert json output from unicode to string
utterances = [[str(file[0]), [str(word).lower() for word in file[1]]] for file in utterances]

# Initialise separate lists for each part

part_utts_list = [[]]*5

# Sort utterances into each correct part list
# Remove <s>, 'sp' and <sil> tokens
# map %hesitation% to 'um'

for item in utterances:
	fileName = item[0]
	sentence = ''
	
	# Determine part
	part_letter = fileName[23]
	part_num = 100
	
	if part_letter == 'A':
		part_num = 0
	elif part_letter == 'B':
		part_num = 1
	elif part_letter == 'C'
		part_num = 2
	elif part_letter == 'D'
		part_num = 3
	elif part_letter == 'E'
		part_num = 4
	else:
		print("part not in range")
		continue

	for word in item[1]:
		if word == '<s>' or word == 'sil' or word == 'sp':
			continue
		elif word == '%hesitation%':
			new_word = 'um'
		elif '%partial%' in word:
			new_word = word[:-10]
		else:
			new_word = word
		sentence = sentence + ' ' + new_word

	part_utts_list[part_num].append([fileName, sentence])


# Output the part lists to separate files
print("Writing to files")

for i in range(5):
	target_file = target_file_prefix + str(i+1) + '.txt'
	with open(target_file, 'w') as f:
		f.truncate(0)
		f.write(json.dumps(part_utts_list[i]))


	
			
