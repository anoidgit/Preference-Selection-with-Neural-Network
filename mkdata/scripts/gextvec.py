#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

from random import random
import h5py
import numpy as np

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
						rsd[wd] = curid
						curid += 1
				else:
					break
	return rsd

def handle(fmap, vecf, rsf, vsize):
	rs={}
	wd = ldvocab(fmap)
	with open(vecf) as f:
		for line in f:
			tmp = line.strip()
			if tmp:
				tmp = tmp.decode("utf-8", "ignore")
				ind = tmp.find(" ")
				w = tmp[:ind]
				if w and w in wd:
					rs[wd[w]]=[float(tmpu) for tmpu in tmp[ind+1:].split()]
	#unkvec = rs.get(1, " ".join([str((random()-0.5)/200) for i in xrange(vsize)]))
	unkvec = rs.get(1, [0.0 for i in xrange(vsize)])
	rsm = []
	for i in xrange(1, len(wd)+1):
		rsm.append(rs.get(i, unkvec))
	rsm = np.array(rsm, dtype=np.float32)
	fwrt = h5py.File(rsf, "w")
	fwrt["emb"] = rsm
	fwrt.close()

if __name__=="__main__":
	handle(sys.argv[1].decode("utf-8"),sys.argv[2].decode("utf-8"),sys.argv[3].decode("utf-8"),int(sys.argv[4].decode("utf-8")))
