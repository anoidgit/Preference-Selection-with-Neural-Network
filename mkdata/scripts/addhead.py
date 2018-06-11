#encoding: utf-8

import sys
from tqdm import tqdm

def handle(fsrc, frs):
	with open(fsrc) as frd:
		with open(frs, "w") as fwrt:
			for line in tqdm(frd):
				tmp = line.strip()
				if tmp:
					tmp = tmp.decode("utf-8")
					v, n = tmp.split()
					tmp = "v_"+v+" n_" +n
					fwrt.write(tmp.encode("utf-8"))
					fwrt.write("\n")

if __name__ == "__main__":
	handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"))
