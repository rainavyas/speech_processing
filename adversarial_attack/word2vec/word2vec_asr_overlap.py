import json

# Load word2vec dict
file_to_read = 'word2vec_emb.txt'
with open(file_to_read, 'r') as f:
	word2vec_dict = json.loads(f.read())
print(len(word2vec_dict.items()))

# Load asr words list
file_to_read = '/home/alta/BLTSpeaking/lms/LM15-grph/wlists/train.lst'
with open(file_to_read, 'r') as f:
	asr_words = f.readlines()
asr_words = [str(word).lower().rstrip() for word in asr_words]

print(len(asr_words))


# Create list to store overlaps
overlaps = []


# Find the overlaps
for asr_word in asr_words:
	if asr_word in word2vec_dict:
		overlaps.append(asr_word)


print(len(overlaps))


#Write to file
target_file = 'test_words.txt'
with open(target_file, 'w') as f:
	f.truncate(0)
	f.write(json.dumps(overlaps))


