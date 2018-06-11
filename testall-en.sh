#!/bin/bash
export mod=$1
export id="180411_nyt_snn_Adam_200_200_pretrained.embedding:true_batchsize:1024_unk:1_minwordfreq:1_mincocfreq:1_cooccur.weight:1_shfc:64"
export srcd=mkdata/data/sp-eng-data1
export rsd=trs/$id
mkdir -p $rsd
export targ=ptb.test.conf.post.txt
bash test.sh $srcd/$targ $rsd/$targ "$1"
export targ=ptb.test.conf.pre.txt
bash test.sh $srcd/$targ $rsd/$targ "$1"
export targ=ptb.test.conf.rand.txt
bash test.sh $srcd/$targ $rsd/$targ "$1"
export targ=ptb.test.pos.post.txt
bash test.sh $srcd/$targ $rsd/$targ "$1"
export targ=ptb.test.pos.pre.txt
bash test.sh $srcd/$targ $rsd/$targ "$1"
export targ=ptb.test.pos.rand.txt
bash test.sh $srcd/$targ $rsd/$targ "$1"
