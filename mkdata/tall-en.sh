#!/bin/bash
export mod=$1
export srcd=data/sp-eng-data1
export rsd=rs/sp-eng-data1/$2
mkdir -p $rsd
export targ=ptb.test.conf.post.txt
pypy scripts/embscore.py $srcd/$targ $rsd/$targ "$1"
export targ=ptb.test.conf.pre.txt
pypy scripts/embscore.py $srcd/$targ $rsd/$targ "$1"
export targ=ptb.test.conf.rand.txt
pypy scripts/embscore.py $srcd/$targ $rsd/$targ "$1"
export targ=ptb.test.pos.post.txt
pypy scripts/embscore.py $srcd/$targ $rsd/$targ "$1"
export targ=ptb.test.pos.pre.txt
pypy scripts/embscore.py $srcd/$targ $rsd/$targ "$1"
export targ=ptb.test.pos.rand.txt
pypy scripts/embscore.py $srcd/$targ $rsd/$targ "$1"
