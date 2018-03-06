#!/bin/bash
export srcd=src
export srctf=sp.train.pd60.txt
export bsize=1024
export allowunk=1
export minfreq=1
export mincocfreq=1
export shfcache=64
export equalW=0
pypy scripts/shuffle.py $srcd/$srctf cache/src.shf
pypy scripts/merge.py $srcd/ cache/valid.txt
pypy scripts/vocab.py cache/src.shf cache/v.vcb cache/n.vcb
rm -fr ../dset.lua
echo -n "verbs=" >> ../dset.lua
pypy scripts/reportvocab.py cache/v.vcb $minfreq >> ../dset.lua
echo -n "nouns=" >> ../dset.lua
pypy scripts/reportvocab.py cache/n.vcb $minfreq >> ../dset.lua
pypy scripts/maptrain.py cache/src.shf ../datasrc/train.h5 cache/v.vcb cache/n.vcb ../dset.lua ntrain $bsize $minfreq $mincocfreq $shfcache $allowunk $equalW
pypy scripts/mapvalid.py cache/valid.txt ../datasrc/valid.h5 cache/v.vcb cache/n.vcb ../dset.lua nvalid $bsize $minfreq $allowunk
