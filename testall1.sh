#!/bin/bash
export mod=$1
export id="180408_mnn_Adam_200_200_pretrained.embedding:false_batchsize:1024_unk:1_minwordfreq:1_mincocfreq:1_cooccur.weight:1_shfc:64"
export srcd=mkdata/data/sp-chi-data1
export rsd=trs/$id
mkdir -p $rsd
export targ=9801.vn.160522.neg.post.txt
bash test.sh $srcd/$targ $rsd/$targ "$1"
export targ=9801.vn.160522.neg.pre.txt
bash test.sh $srcd/$targ $rsd/$targ "$1"
export targ=9801.vn.160522.neg.rand.txt
bash test.sh $srcd/$targ $rsd/$targ "$1"
export targ=9801.vn.160522.pos.post.txt
bash test.sh $srcd/$targ $rsd/$targ "$1"
export targ=9801.vn.160522.pos.pre.txt
bash test.sh $srcd/$targ $rsd/$targ "$1"
export targ=9801.vn.160522.pos.rand.txt
bash test.sh $srcd/$targ $rsd/$targ "$1"
