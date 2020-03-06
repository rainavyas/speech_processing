#!/bin/tcsh
#$ -S /bin/tcsh


# The plp making tool requires the coding file to be stored as $TSET.plp in ./lib/coding/
# The tool also requires a configuration file: ./lib/cfgs/plp.cfg

#set TSET='BLXXXgrd02_original'
set TSET='BLXXXgrd02_altered'

# Delete all current plp vector files
rm /home/alta/BLTSpeaking/grd-graphemic-vr313/speech_processing/merger/adversarial/evolutionary/data/BLXXXgrd02_altered_plp/*.plp

set jwaitplp=`/home/alta/CHILD_Shared/convert/local/tools/mkplp $TSET lib plp`

echo $jwaitplp > killall.$TSET
