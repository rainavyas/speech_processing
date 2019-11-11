#!/bin/bash
#$ -S /bin/bash

PART=$1
SEED=$2
NUM=$3
INDEX=$4


#export PATH=/home/miproj/4thyr.oct2019/vr313/.local/lib/python3.7/site-packages:$PATH
#export PYTHONPATH=/home/miproj/4thyr.oct2019/vr313/.local/lib/python3.7/site-packages

source /home/alta/BLTSpeaking/grd-graphemic-vr313/speech_processing/venv/bin/activate

python3.6 word_search.py --part=$PART --seed=$SEED --num=$NUM --index=$INDEX

