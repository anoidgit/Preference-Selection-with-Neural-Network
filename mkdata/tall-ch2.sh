#!/bin/bash
export mod=$1
export srcd=data/sp-chi-data2
export rsd=rs/sp-chi-data2
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
