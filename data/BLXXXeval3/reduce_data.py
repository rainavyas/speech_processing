import json

select_num = 20

file_prefix = 'useful_part'

utts_per_part = [6, 8, 1, 1, 5]

for i in range(5):
	utt_num = utts_per_part[i]
	input_file = file_prefix + str(i+1)+'.txt'
	
	# Read in the part data file
	with open(input_file, 'r') as f:
		utterances = json.loads(f.read())

	
	# Select only certain number of utterances
	to_select = select_num * utt_num

	selected_utts = utterances[:to_select]
	
	# Write to file
	target_file = file_prefix + str(i+1)+'_reduced.txt'
	with open(target_file, 'w') as f:
		f.truncate(0)
		f.write(json.dumps(selected_utts))



