#encoding: utf-8

import sys
import h5py
import numpy as np
from random import shuffle
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

def ldata(fsrc, vd, nd, bsize, nshuffle = 64, cfld = 1, allowunk = True, equally = False):
	cache = []
	lim = bsize * nshuffle
	with open(fsrc) as frd:
		for line in tqdm(frd):
			tmp = line.strip()
			if tmp:
				tmp = tmp.decode("utf-8")
				v, n, f = tmp.split()
				vid = vd.get(v, 1)
				nid = nd.get(n, 1)
				if allowunk or (vid!=1 and nid!=1):
					if equally:
						cache.append([vid, nid])
					else:
						for i in xrange(int(f)):
							cache.append([vid, nid])
					if len(cache) >= lim:
						ctw = cache[:lim]
						shuffle(ctw)
						sind = 0
						eind = nshuffle
						for i in xrange(nshuffle):
							yield(np.array(ctw[sind:eind], dtype=np.int32))
							sind = eind
							eind += nshuffle
						cache = cache[lim:]
	if cache:
		shuffle(cache)
		nw = int(floor(len(cache)/bsize))
		if nw > 0:
			sind = 0
			eind = nshuffle
			for i in xrange(nw):
				yield(np.array(cache[sind:eind], dtype=np.int32))
				sind = eind
				eind += nshuffle
			yield(np.array(cache[sind:], dtype=np.int32))
			cache = []
		else:
			yield(np.array(cache, dtype=np.int32))
			cache = []

def handle(fsrc, frs, fdv, fdn, flog, fattr, bsize, fld = 1, cfld = 1, nshuffle = 64, allowunk = True, equally = False):
	vd, nd = ldvocab(fdv, fld), ldvocab(fdn, fld)
	fwrt = h5py.File(frs, "w")
	curid = 1
	for du in ldata(fsrc, vd, nd, bsize, nshuffle, cfld, allowunk, equally):
		fwrt[str(curid)] = du
		curid += 1
	fwrt.close()
	with open(flog, "a") as f:
		tmp=fattr+"="+str(curid-1)+"\n"
		f.write(tmp.encode("utf-8"))

if __name__ == "__main__":
	if len(sys.argv)>12:
		handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"), sys.argv[3].decode("utf-8"), sys.argv[4].decode("utf-8"), sys.argv[5].decode("utf-8"), sys.argv[6].decode("utf-8"), int(sys.argv[7].decode("utf-8")), int(sys.argv[8].decode("utf-8")), int(sys.argv[9].decode("utf-8")), int(sys.argv[10].decode("utf-8")), bool(int(sys.argv[11].decode("utf-8"))), bool(int(sys.argv[12].decode("utf-8"))))
	elif len(sys.argv)==12:
		handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"), sys.argv[3].decode("utf-8"), sys.argv[4].decode("utf-8"), sys.argv[5].decode("utf-8"), sys.argv[6].decode("utf-8"), int(sys.argv[7].decode("utf-8")), int(sys.argv[8].decode("utf-8")), int(sys.argv[9].decode("utf-8")), int(sys.argv[10].decode("utf-8")), bool(int(sys.argv[11].decode("utf-8"))))
	elif len(sys.argv)==11:
		handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"), sys.argv[3].decode("utf-8"), sys.argv[4].decode("utf-8"), sys.argv[5].decode("utf-8"), sys.argv[6].decode("utf-8"), int(sys.argv[7].decode("utf-8")), int(sys.argv[8].decode("utf-8")), int(sys.argv[9].decode("utf-8")), int(sys.argv[10].decode("utf-8")))
	elif len(sys.argv)==10:
		handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"), sys.argv[3].decode("utf-8"), sys.argv[4].decode("utf-8"), sys.argv[5].decode("utf-8"), sys.argv[6].decode("utf-8"), int(sys.argv[7].decode("utf-8")), int(sys.argv[8].decode("utf-8")), int(sys.argv[9].decode("utf-8")))
	elif len(sys.argv)==9:
		handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"), sys.argv[3].decode("utf-8"), sys.argv[4].decode("utf-8"), sys.argv[5].decode("utf-8"), sys.argv[6].decode("utf-8"), int(sys.argv[7].decode("utf-8")), int(sys.argv[8].decode("utf-8")))
	else:
		handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"), sys.argv[3].decode("utf-8"), sys.argv[4].decode("utf-8"), sys.argv[5].decode("utf-8"), sys.argv[6].decode("utf-8"), int(sys.argv[7].decode("utf-8")))
