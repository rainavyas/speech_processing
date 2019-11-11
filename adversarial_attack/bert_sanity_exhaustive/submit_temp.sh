#!/bin/tcsh

set ALLARGS = ($*)

# Check Number of Args
if ( $#argv != 3) then
   echo "Usage: $0 part seed num_words_check log"
   echo "  e.g: $0 3 1 100"
   exit 100
endif

set PART=$1
set SEED=$2
set NUM=$3

set SCRIPT=run_script.sh


set INDEX=1

while ($INDEX <= 10)

	qsub -P esol -l ubuntu=1 -l qp=low -l osrel="*" -o LOGs/run.txt -j y $SCRIPT $PART $SEED $NUM $INDEX
	@ INDEX++
end


