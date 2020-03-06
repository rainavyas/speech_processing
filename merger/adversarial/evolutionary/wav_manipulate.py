import numpy as np
from scipy.io import wavfile
from random import seed
from random import random

def show_info(aname, a):
	print("Array", aname)
	print("shape:", a.shape)
	print ("dtype:", a.dtype)
	print ("min, max:", a.min(), a.max())



def add_periodic_noise(signal, noise_unit)




rate, data = wavfile.read('data/BLXXXgrd02_original_wav/SRJMZ8RNSX/SRJMZ8RNSX_SA_03.wav')

show_info("data", data)



# Create 8-frame noise randomly
NUM_FRAMES = 8
# Allow noise values to be up to 1000 in amplitude
NOISE_AMP = 1000


seed(1)

noise_vals = [0]*NUM_FRAMES
noise_vals = [NOISE_AMP*random() for i in noise_vals]

print(noise_vals)





