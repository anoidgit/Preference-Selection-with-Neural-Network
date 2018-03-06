#!/bin/bash
export srcf=$1
export wdir=cache/wvec

export ef=$wdir/exp.txt

mkdir -p $wdir
pypy scripts/expand.py $srcf $ef

export usecbow=0
export VECTOR_SIZE=64
export VOCAB_MIN_COUNT=1
export MAX_ITER=256
export WINDOW_SIZE=1
export NUM_THREADS=8
export BINARY=0
export SAVE_FILE=$wdir/wvec_$VECTOR_SIZE.$VOCAB_MIN_COUNT.$MAX_ITER.$WINDOW_SIZE

echo "$ word2vec -output $SAVE_FILE -threads $NUM_THREADS -train $ef -iter $MAX_ITER -size $VECTOR_SIZE -binary $BINARY -cbow $usecbow -window $WINDOW_SIZE -min-count $VOCAB_MIN_COUNT"
word2vec -output $SAVE_FILE -threads $NUM_THREADS -train $ef -iter $MAX_ITER -size $VECTOR_SIZE -binary $BINARY -cbow $usecbow -window $WINDOW_SIZE -min-count $VOCAB_MIN_COUNT
