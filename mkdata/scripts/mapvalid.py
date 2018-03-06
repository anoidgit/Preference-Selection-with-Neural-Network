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
						rsd[wd] = curid
						curid += 1
				else:
					break
	return rsd

def ldata(fsrc, vd, nd, bsize, allowunk = True):
	cache = []
	with open(fsrc) as frd:
		for line in tqdm(frd):
			tmp = line.strip()
			if tmp:
				tmp = tmp.decode("utf-8")
				v, pn, nn = tmp.split()
				vid = vd.get(v, 1)
				npid = nd.get(pn, 1)
				nnid = nd.get(nn, 1)
				if allowunk or (vid!=1 and npid!=1 and nnid!=1):
					cache.append([vid, npid, nnid])
				if len(cache) >= bsize:
					ctw = cache[:bsize]
					yield(np.array(ctw, dtype=np.int32))
					cache = cache[bsize:]
	if cache:
		nw = int(floor(len(cache)/bsize))
		if nw > 0:
			sind = 0
			eind = bsize
			for i in xrange(nw):
				yield(np.array(cache[sind:eind], dtype=np.int32))
				sind = eind
				eind += bsize
			yield(np.array(cache[sind:], dtype=np.int32))
			cache = []
		else:
			yield(np.array(cache, dtype=np.int32))
			cache = []

def handle(fsrc, frs, fdv, fdn, flog, fattr, bsize, fld = 1, allowunk = True):
	vd, nd = ldvocab(fdv, fld), ldvocab(fdn, fld)
	fwrt = h5py.File(frs, "w")
	curid = 1
	for du in ldata(fsrc, vd, nd, bsize, allowunk):
		fwrt[str(curid)] = du
		curid += 1
	fwrt.close()
	with open(flog, "a") as f:
		tmp=fattr+"="+str(curid-1)+"\n"
		f.write(tmp.encode("utf-8"))

if __name__ == "__main__":
	if len(sys.argv)>9:
		handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"), sys.argv[3].decode("utf-8"), sys.argv[4].decode("utf-8"), sys.argv[5].decode("utf-8"), sys.argv[6].decode("utf-8"), int(sys.argv[7].decode("utf-8")), int(sys.argv[8].decode("utf-8")), bool(int(sys.argv[9].decode("utf-8"))))
	elif len(sys.argv)==9:
		handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"), sys.argv[3].decode("utf-8"), sys.argv[4].decode("utf-8"), sys.argv[5].decode("utf-8"), sys.argv[6].decode("utf-8"), int(sys.argv[7].decode("utf-8")), int(sys.argv[8].decode("utf-8")))
	else:
		handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"), sys.argv[3].decode("utf-8"), sys.argv[4].decode("utf-8"), sys.argv[5].decode("utf-8"), sys.argv[6].decode("utf-8"), int(sys.argv[7].decode("utf-8")))
