#!/bin/bash
export mod=$1
export srcd=src
export rsd=rs/wrs
mkdir -p $rsd
export targ=9801.post.neg.txt
pypy scripts/embscore.py $srcd/$targ $rsd/$targ "$1"
export targ=9801.post.pos.txt
pypy scripts/embscore.py $srcd/$targ $rsd/$targ "$1"
export targ=9801.pre.neg.txt
pypy scripts/embscore.py $srcd/$targ $rsd/$targ "$1"
export targ=9801.pre.pos.txt
pypy scripts/embscore.py $srcd/$targ $rsd/$targ "$1"
export targ=9801.rand.neg.txt
pypy scripts/embscore.py $srcd/$targ $rsd/$targ "$1"
export targ=9801.rand.pos.txt
pypy scripts/embscore.py $srcd/$targ $rsd/$targ "$1"
