#!/bin/bash
export mod=$1
export srcd=data/sp-chi-data1
export rsd=rs/sp-chi-data1
mkdir -p $rsd
export targ=9801.vn.160522.neg.post.txt
pypy scripts/embscore.py $srcd/$targ $rsd/$targ "$1"
export targ=9801.vn.160522.neg.pre.txt
pypy scripts/embscore.py $srcd/$targ $rsd/$targ "$1"
export targ=9801.vn.160522.neg.rand.txt
pypy scripts/embscore.py $srcd/$targ $rsd/$targ "$1"
export targ=9801.vn.160522.pos.post.txt
pypy scripts/embscore.py $srcd/$targ $rsd/$targ "$1"
export targ=9801.vn.160522.pos.pre.txt
pypy scripts/embscore.py $srcd/$targ $rsd/$targ "$1"
export targ=9801.vn.160522.pos.rand.txt
pypy scripts/embscore.py $srcd/$targ $rsd/$targ "$1"
