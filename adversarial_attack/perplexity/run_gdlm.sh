#!/bin/bash
#$ -S /bin/bash






# Define the binary file
RNNLMBIN=/home/alta/BLTSpeaking/exp-kmk-v2/RNNLM/RNNLM.cued/src.v1.1.local/rnnlm.cued.v1.1

# Define the model
RNNMODEL=/home/alta/BLTSpeaking/exp-lw519/GDLMs/GDRNNLM2/RNN_weight.OOS.cuedrnnlm/train_LM.wgt

# Define input word list
RNNINWLIST=/home/alta/BLTSpeaking/exp-lw519/GDLMs/GDRNNLM2/RNN_weight.OOS.cuedrnnlm/train_LM.wgt.input.wlist.index

# Define output word list
RNNOUWLIST=/home/alta/BLTSpeaking/exp-lw519/GDLMs/GDRNNLM2/RNN_weight.OOS.cuedrnnlm/train_LM.wgt.output.wlist.index

# Define Language Model Feature File
RNNFEAFILE=/home/alta/BLTSpeaking/exp-lw519/GDLMs/data/fea.mat

# Give the file with list of all dat files to run GDLM on
TFLIST=tflist.txt

# Define where to save the output
OUTPUT=results/rnnlml_ppl_out.txt




# run GDLM over each word's .dat file in flist of all dat files
$RNNLMBIN -ppl -readmodel $RNNMODEL -inputwlist $RNNINWLIST -outputwlist $RNNOUWLIST -feafile $RNNFEAFILE -testflist $TFLIST > $OUTPUT
