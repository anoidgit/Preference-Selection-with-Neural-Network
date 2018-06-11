#!/bin/bash
export srcd=data/sp-eng-data1
export srctf=$1
export bsize=1024
export allowunk=1
export minfreq=1
export mincocfreq=1
export shfcache=64
export equalW=0
pypy scripts/shuffle.py $srctf cache/src.shf
pypy scripts/mergen.py $srcd/ cache/valid.txt
pypy scripts/vocab.py cache/src.shf cache/v.vcb cache/n.vcb
pypy scripts/gextvec.py cache/v.vcb data/glove.6B.200d.txt ../datasrc/v.h5 200
pypy scripts/gextvec.py cache/n.vcb data/glove.6B.200d.txt ../datasrc/n.h5 200
rm -fr ../dset.lua
echo -n "verbs=" >> ../dset.lua
pypy scripts/reportvocab.py cache/v.vcb $minfreq >> ../dset.lua
echo -n "nouns=" >> ../dset.lua
pypy scripts/reportvocab.py cache/n.vcb $minfreq >> ../dset.lua
pypy scripts/maptrain.py cache/src.shf ../datasrc/train.h5 cache/v.vcb cache/n.vcb ../dset.lua ntrain $bsize $minfreq $mincocfreq $shfcache $allowunk $equalW
pypy scripts/mapvalid.py cache/valid.txt ../datasrc/valid.h5 cache/v.vcb cache/n.vcb ../dset.lua nvalid $bsize $minfreq $allowunk
