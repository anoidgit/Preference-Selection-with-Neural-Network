#!/bin/bash
export mod=$1
export id="180409_d2_snn_Adam_200_200_pretrained.embedding:false_batchsize:1024_unk:1_minwordfreq:1_mincocfreq:1_cooccur.weight:1_shfc:64"
export srcd=mkdata/data/sp-chi-data2
export rsd=trs/$id
mkdir -p $rsd
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
