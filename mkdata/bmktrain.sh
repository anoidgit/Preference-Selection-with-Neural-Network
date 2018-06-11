#!/bin/bash
export srcd=data/sp-chi-data1
export srctf=pd-pwb.obj.train.30.5.final.txt
export bsize=1024
export allowunk=1
export minfreq=1
export mincocfreq=1
export shfcache=64
export equalW=0
pypy scripts/shuffle.py $srcd/$srctf cache/src.shf
pypy scripts/merge.py $srcd/ cache/valid.txt
pypy scripts/bvocab.py cache/src.shf cache/v.vcb cache/n.vcb
pypy scripts/extvec.py cache/v.vcb data/pd60vec200.2.txt ../datasrc/v.h5 200
pypy scripts/extvec.py cache/n.vcb data/pd60vec200.2.txt ../datasrc/n.h5 200
rm -fr ../dset.lua
echo -n "verbs=" >> ../dset.lua
pypy scripts/reportvocab.py cache/v.vcb $minfreq >> ../dset.lua
echo -n "nouns=" >> ../dset.lua
pypy scripts/reportvocab.py cache/n.vcb $minfreq >> ../dset.lua
pypy scripts/bmaptrain.py cache/src.shf ../datasrc/train.h5 cache/v.vcb cache/n.vcb ../dset.lua ntrain $bsize $minfreq $mincocfreq $shfcache $allowunk $equalW
pypy scripts/mapvalid.py cache/valid.txt ../datasrc/valid.h5 cache/v.vcb cache/n.vcb ../dset.lua nvalid $bsize $minfreq $allowunk
