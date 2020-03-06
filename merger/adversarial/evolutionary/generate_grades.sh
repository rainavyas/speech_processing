#!/bin/tcsh
#$ -S /bin/tcsh


python2 /home/alta/BLTSpeaking/exp-kk492/alta_tools/tf_grader/deep_pron_grade.py -FEATS deep_pron -NAME pron -AGG_ARCH att -MAX_SPEAKER_LEN 3200 -LOAD_PATHS /home/alta/BLTSpeaking/grd-kk492/GKTS4-D3/grader/BLXXXgrd02/deep_pron/grader-300/model -DROPOUT 1.0 -SEED 300 -BATCH_NORM True -DATA plp2pkl/BLXXXgrd02_altered.pkl -MODE eval -SAVE_PATH /home/alta/BLTSpeaking/grd-kk492/plp13/GKTS4-D3/grader/BLXXXgrd02/deep_pron/grader-300/model -OUT_PATH grader_results/BLXXXgrd02_altered/deep_pron/grader-300
