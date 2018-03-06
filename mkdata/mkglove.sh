#!/bin/bash
export srcf=$1
export wdir=cache/glove

export ef=$wdir/exp.txt

mkdir -p $wdir
pypy scripts/expand.py $srcf $ef

export VECTOR_SIZE=64
export VOCAB_MIN_COUNT=1
export MAX_ITER=256
export WINDOW_SIZE=1
export NUM_THREADS=8
export BINARY=0
export VOCAB_FILE=$wdir/vocab.txt
export COOCCURRENCE_FILE=$wdir/cooccurrence.bin
export COOCCURRENCE_SHUF_FILE=$wdir/cooccurrence.shuf.bin
export SAVE_FILE=$wdir/glove_$VECTOR_SIZE.$VOCAB_MIN_COUNT.$MAX_ITER.$WINDOW_SIZE
export VERBOSE=2
export MEMORY=8.0
export X_MAX=100

echo "$ vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $ef > $VOCAB_FILE"
vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $ef > $VOCAB_FILE
echo "$ cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $ef > $COOCCURRENCE_FILE"
cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $ef > $COOCCURRENCE_FILE
echo "$ shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
echo "$ glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
