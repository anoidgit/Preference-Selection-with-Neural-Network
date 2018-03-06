#!/bin/bash
export srcf=../$1
export minfreq=1
export bsize=256
pypy scripts/maptst.py $srcf ../datasrc/test.h5 cache/v.vcb cache/n.vcb ../tdset.lua ntest $bsize $minfreq
