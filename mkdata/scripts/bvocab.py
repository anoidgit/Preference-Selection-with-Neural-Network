#encoding: utf-8

import sys

def extvocab(fsrc):
	vd = {}
	nd = {}
	with open(fsrc) as f:
		for line in f:
			tmp = line.strip()
			if tmp:
				v, n = tmp.decode("utf-8").split("	")
				vd[v] = vd.get(v, 0) + 1
				nd[n] = nd.get(n, 0) + 1
	return vd, nd

def sortvocab(vin):
	rsd = {}
	for k, v in vin.iteritems():
		if not v in rsd:
			rsd[v] = set([k])
		else:
			rsd[v].add(k)
	return rsd

def writevocab(frs, vw):
	with open(frs, "w") as f:
		l = vw.keys()
		l.sort(reverse=True)
		for lu in l:
			tmp = [str(lu)]
			tmp.extend(list(vw[lu]))
			tmp = " ".join(tmp)
			f.write(tmp.encode("utf-8"))
			f.write("\n")

def handle(fsrc, fv, fn):
	vd, nd = extvocab(fsrc)
	writevocab(fv, sortvocab(vd))
	writevocab(fn, sortvocab(nd))

if __name__ == "__main__":
	handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"), sys.argv[3].decode("utf-8"))
