#!/bin/tcsh
#$ -S /bin/tcsh

# Make PLP flists
# a. fix lengths in individual files 
# b. generate files for merging
# c. merge files using HCopy

#set verbose

set ALLARGS=($*)

set CHANGED

# Check Number of Args 

if ( $#argv != 1) then
   echo "Usage: $0  tset"
   echo " e.g.: $0  LINSKall01"
   exit 100
endif

set TSET=$1
set CODE=lib/coding/$TSET.plp
if ( ! -f $CODE ) then
    echo "ERROR: coding script not found: $CODE"
    exit 100
endif

if ( ! -d CMDs ) mkdir -p CMDs
set CMDFILE=CMDs/mkplpflist.cmds
echo "------------------------------------" >> $CMDFILE
echo "$0 $ALLARGS" >> $CMDFILE
echo "------------------------------------" >> $CMDFILE

mkdir -p LOGs
set LIBDIR=lib

set pardir = plp
set param = plp
set WKDIR=$LIBDIR/flists.$pardir
mkdir -p $WKDIR

# 1) generate the individual param script file
set INDFILE=$WKDIR/${TSET}-base.ind.$param

# Generate a file list with lengths [0,0]
echo "awk -f local/lib/awks/scpmap-raw.awk lib/coding/$TSET.$pardir > $INDFILE"
awk -f local/lib/awks/scpmap-raw.awk lib/coding/$TSET.$pardir | sort -n > $INDFILE

if ( ! -f $INDFILE ) then
    echo "ERROR: file not created: $INDFILE"
    exit 100
endif

# Add the length and rename (!) the old file .org
./local/tools/addlength $INDFILE $param
awk -f local/lib/awks/scpmaptime.awk $INDFILE.bas > $INDFILE.scp

# 2) Handle any end effects on the PLP files (linear xform bug in HTK)
./base/tools/fixlength $INDFILE.scp $param

mkdir -p $LIBDIR/flists.$pardir/work
mv $LIBDIR/flists.$pardir/${TSET}-base.ind.${param}* $LIBDIR/flists.$pardir/work
cp $LIBDIR/flists.$pardir/work/${TSET}-base.ind.${param}.scp $LIBDIR/flists.$pardir/$TSET.scp

mv $LIBDIR/flists.$pardir/$TSET.scp $LIBDIR/flists.$pardir/work
sort -n $LIBDIR/flists.$pardir/work/$TSET.scp > $LIBDIR/flists.$pardir/$TSET.scp


