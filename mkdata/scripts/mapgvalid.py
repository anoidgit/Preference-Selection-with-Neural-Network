#encoding: utf-8

import sys
from tqdm import tqdm

def ldvocab(fsrc, minfreq = 0):
	rsd = {}
	curid = 0
	with open(fsrc) as frd:
		for line in frd:
			tmp = line.strip()
			if tmp:
				wd, freq = tmp.decode("utf-8").split()
				freq = int(freq)
				if freq >= minfreq:
					rsd[wd] = str(curid)
					curid += 1
	print("Load vocab with size: "+str(curid))
	return rsd

def handle(fsrc, frs, fd, minfreq = 0):
	rsd = ldvocab(fd, minfreq)
	with open(fsrc) as frd:
		with open(frs, "w") as fwrt:
			for line in frd:
				tmp = line.strip()
				if tmp:
					tmp = tmp.decode("utf-8")
					v, np, nn= tmp.split()
					vid = rsd.get("v_"+v, "-1")
					npid = rsd.get("n_"+np, "-1")
					nnid = rsd.get("n_"+nn, "-1")
					tmp = ",".join((vid, npid, nnid,))
					fwrt.write(tmp.encode("utf-8"))
					fwrt.write("\n")

if __name__ == "__main__":
	handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"), sys.argv[3].decode("utf-8"), int(sys.argv[4].decode("utf-8")))
