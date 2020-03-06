import os
import scandir

base_path = "/home/alta/BLTSpeaking/grd-graphemic-vr313/speech_processing/merger/adversarial/evolutionary/data/"
#wav_folder = "BLXXXgrd02_original_wav"
#plp_folder = "BLXXXgrd02_original_plp"
wav_folder = "BLXXXgrd02_altered_wav"
plp_folder = "BLXXXgrd02_altered_plp"


# Use existing coding file to get map from utterance id to CUED id
existing_coding_file = "/home/alta/BLTSpeaking/convert-v2/1/lib/coding/BLXXXgrd02.plp"

with open(existing_coding_file, 'r') as f:
	lines = f.readlines()


id_map = {}
for line in lines:
	utt_id = str(line[62:78])
	cued_id = line[-40:-5]
	id_map[utt_id] = cued_id
	




# get list of subfolders in the wav folder
#wav_sub_folders = [f.name for f in os.scandir(base_path+wav_folder) if f.is_dir()]
wav_sub_folders = [f.name for f in scandir.scandir(base_path+wav_folder) if f.is_dir()]


# Write wav file to plp file into coding file line by line
#coding_file = open("lib/coding/BLXXXgrd02_original.plp", "a")
coding_file = open("lib/coding/BLXXXgrd02_altered.plp", "a")
coding_file.truncate(0)

for sub_folder in wav_sub_folders:
	#wav_files = [f.name for f in os.scandir(base_path+wav_folder+"/"+sub_folder)]
	wav_files = [f.name for f in scandir.scandir(base_path+wav_folder+"/"+sub_folder)]


	for wav_file in wav_files:
		
		file_name = wav_file[:-4]
		cued_id = id_map[file_name]
		wav_file_path = base_path+wav_folder+"/"+sub_folder+"/"+wav_file
		plp_file_path = base_path+plp_folder+"/"+cued_id+".plp"
		
		entry = wav_file_path + " " + plp_file_path+"\n"

		# Write entry into coding file
		coding_file.write(entry)

coding_file.close()

