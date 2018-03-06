#!/bin/bash
export mod=$1
export srcd=mkdata/src
export rsd=trs
export targ=9801.post.neg.txt
bash test.sh $srcd/$targ $rsd/$targ "$1"
export targ=9801.post.pos.txt
bash test.sh $srcd/$targ $rsd/$targ "$1"
export targ=9801.pre.neg.txt
bash test.sh $srcd/$targ $rsd/$targ "$1"
export targ=9801.pre.pos.txt
bash test.sh $srcd/$targ $rsd/$targ "$1"
export targ=9801.rand.neg.txt
bash test.sh $srcd/$targ $rsd/$targ "$1"
export targ=9801.rand.pos.txt
bash test.sh $srcd/$targ $rsd/$targ "$1"
