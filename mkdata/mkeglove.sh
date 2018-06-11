#!/bin/bash
export srcf=$1
export srcd=$2
export wdir=cache/glove
export rsdir=modrs

export ef=$wdir/exp.txt

mkdir -p $wdir
pypy scripts/expand.py $srcf $ef

export VECTOR_SIZE=200
export VOCAB_MIN_COUNT=1
export MAX_ITER=1024
export WINDOW_SIZE=1
export X_MAX=100
export NUM_THREADS=8
export BINARY=0
export VOCAB_FILE=$wdir/vocab.txt
export COOCCURRENCE_FILE=$wdir/cooccurrence.bin
export COOCCURRENCE_SHUF_FILE=$wdir/cooccurrence.shuf.bin
export SAVE_FILE=$rsdir/glove_$3.$VECTOR_SIZE.$VOCAB_MIN_COUNT.$MAX_ITER.$WINDOW_SIZE.$X_MAX
export VERBOSE=2
export MEMORY=8.0

echo "$ vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $ef > $VOCAB_FILE"
vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $ef > $VOCAB_FILE
pypy scripts/mergen.py $srcd/ cache/valid.txt
pypy scripts/mapgvalid.py cache/valid.txt $wdir/valid.txt $VOCAB_FILE $VOCAB_MIN_COUNT
echo "$ cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $ef > $COOCCURRENCE_FILE"
cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $ef > $COOCCURRENCE_FILE
echo "$ shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
echo "$ vglove/vg -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
vglove/vg -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
