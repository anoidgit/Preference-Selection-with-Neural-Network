#encoding: utf-8

import sys

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

def handle(fvocab, minf=1):
	print(len(ldvocab(fvocab, minf)))

if __name__ == "__main__":
	if len(sys.argv)>2:
		handle(sys.argv[1].decode("utf-8"), int(sys.argv[2].decode("utf-8")))
	else:
		handle(sys.argv[1].decode("utf-8"))
