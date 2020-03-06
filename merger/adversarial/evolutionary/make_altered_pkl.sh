#!/bin/tcsh
#$ -S /bin/tcsh


 python2 plp2pkl/plp2pkl.py -WLAB /home/alta/BLTSpeaking/exp-kk492/GKTS4-D3/align-BLXXXgrd02-rnnlm.mpe/lib/wlabs/train.mlf -MLAB /home/alta/BLTSpeaking/exp-kk492/GKTS4-D3/align-BLXXXgrd02-rnnlm.mpe/lib/mlabs/train.mlf -SCP /home/alta/BLTSpeaking/grd-graphemic-vr313/speech_processing/merger/adversarial/evolutionary/lib/flists.plp/BLXXXgrd02_altered.scp -TSET BLXXXgrd02_altered -ALPHABET arpabet -MSGPACK
