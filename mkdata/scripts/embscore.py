#encoding: utf-8

import sys

from math import sqrt

def ldemb(fsrc):
	rsd = {}
	with open(fsrc) as frd:
		for line in frd:
			tmp = line.strip()
			if tmp:
				tmp = tmp.decode("utf-8").split()
				rsd[tmp[0]] = [float(i) for i in tmp[1:]]
	return rsd

def sum_vec(vl):
	sum = 0
	for vu in vl:
		sum += vu
	return sum

def mul_vec(v1, v2):
	return [v1u * v2u for v1u, v2u in zip(v1, v2)]

def add_vec(v1, v2):
	return [v1u + v2u for v1u, v2u in zip(v1, v2)]

def dot_vec(v1, v2):
	return sum_vec(mul_vec(v1, v2))

def cos_vec(v1, v2):
	nv1 = dot_vec(v1, v1)
	nv2 = dot_vec(v2, v2)
	d = dot_vec(v1, v2)
	return d / sqrt(nv1 * nv2)

def sim(v1, v2):
	return cos_vec(v1, v2)

def handle(fsrc, frs, fembd):
	emb = ldemb(fembd)
	unk = emb.get("<unk>")
	with open(fsrc) as frd:
		with open(frs, "w") as fwrt:
			for line in frd:
				tmp = line.strip()
				if tmp:
					v, n = tmp.decode("utf-8").split()
					rs = str(sim(emb.get("v_"+v, unk), emb.get("n_"+n, unk)))
					fwrt.write(rs.encode("utf-8"))
				fwrt.write("\n")

if __name__ == "__main__":
	handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"), sys.argv[3].decode("utf-8"))
