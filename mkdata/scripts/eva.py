#encoding: utf-8

import sys

def handle(fpos, fneg):
	tot = 0
	crc = 0
	with open(fpos) as fp:
		with open(fneg) as fn:
			for ps, ns in zip(fp, fn):
				pt = ps.strip()
				if pt:
					ps = float(pt.decode("utf-8"))
					ns = float(ns.strip().decode("utf-8"))
					if ps > ns:
						crc += 1
					tot += 1
	return float(crc)/float(tot)

if __name__ == "__main__":
	print(handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8")))
