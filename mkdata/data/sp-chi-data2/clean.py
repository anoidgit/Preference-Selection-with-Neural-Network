#encoding: utf-8

import os

col = set()

for root, dirs, files in os.walk("."):
	for f in files:
		if f.startswith("9801"):
			with open(os.path.join(root, f)) as frd:
				for line in frd:
					tmp = line.strip()
					if tmp:
						tmp = tmp.decode("utf-8")
						if not tmp in col:
							col.add(tmp)

with open("sp.train.pd60.txt") as frd:
	with open("train.txt", "w") as fwrt:
		for line in frd:
			tmp = line.strip()
			if tmp:
				tmp = tmp.decode("utf-8")
				v = tmp[:tmp.rfind("\t")]
				if not v in col:
					fwrt.write(tmp.encode("utf-8"))
					fwrt.write("\n")

