#encoding: utf-8

import sys
import h5py
import numpy as np
from math import floor
from tqdm import tqdm

def ldvocab(fsrc, minf=1):
	rsd = {"<unk>":1}
	curid = 2
	with open(fsrc) as frd:
		for line in frd:
			tmp = line.strip()
			if tmp:
				tmp = tmp.decode("utf-8").split()
				f = int(tmp[0])
				if f >= minf:
					for wd in tmp[1:]:
						rsd[str(curid)] = wd
						curid += 1
				else:
					break
	return rsd

def handle(fsrc, wid, fld = 1):
	vd = ldvocab(fsrc, fld)
	print(vd.get(wid, "unk"))

if __name__ == "__main__":
	if len(sys.argv)>3:
		handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"), int(sys.argv[3].decode("utf-8")))
	else:
		handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"))
