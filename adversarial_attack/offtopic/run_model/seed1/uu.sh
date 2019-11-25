#!/bin/bash
#$ -S /bin/bash

export LD_LIBRARY_PATH="/home/mifs/am969/cuda-8.0/lib64:${LD_LIBRARY_PATH}"

export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE

source /home/alta/BLTspeaking/grd-graphemic-vr313/speech_processing/venv/bin/activate

../../offtopic_model/step_test_hatm.py ../../data/tfrecords/relevance.test.tfrecords relevance_results --epoch 3
