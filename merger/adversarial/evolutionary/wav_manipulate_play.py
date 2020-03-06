import numpy as np
from scipy.io import wavfile
from random import seed
from random import random

def show_info(aname, a):
	print("Array", aname)
	print("shape:", a.shape)
	print ("dtype:", a.dtype)
	print ("min, max:", a.min(), a.max())



def add_periodic_noise(signal, noise_unit):
	num_repeats_required = signal.shape[0]/len(noise_unit)
	extra_vals_required = signal.shape[0] - (num_repeats_required*len(noise_unit))

	extra_vals = noise_unit[:extra_vals_required]	

	periodic_noise = []
	for count in range(num_repeats_required):
		periodic_noise += noise_unit
	
	periodic_noise += extra_vals
	periodic_noise = np.array(periodic_noise)

	print(periodic_noise.shape)

	# Add the noise to the signal
	combined = np.add(signal, periodic_noise)

	return combined
	



rate, data = wavfile.read('data/BLXXXgrd02_original_wav/SRJMZ8RNSX/SRJMZ8RNSX_SA_03.wav')

show_info("data", data)



# Create 8-frame noise randomly
NUM_FRAMES = 8
# Allow noise values to be up to 1000 in amplitude
NOISE_AMP = 1000


seed(1)

noise_vals = [0]*NUM_FRAMES
noise_vals = [NOISE_AMP*random() - NOISE_AMP/2 for i in noise_vals]


combined = add_periodic_noise(data, noise_vals)

show_info("combined", combined)




