#encoding: utf-8

import sys
import os

def handle(srcd, rsf):
	with open(rsf, "w") as fwrt:
		for root, dirs, files in os.walk(srcd):
			for file in files:
				if file.find("pos")>0:
					with open(os.path.join(root, file)) as fpos:
						with open(os.path.join(root, file.replace(".pos.", ".neg."))) as fneg:
							for pd, nd in zip(fpos, fneg):
								tmpp = pd.strip()
								if tmpp:
									vp, np = tmpp.decode("utf-8").split()
									vn, nn = nd.strip().decode("utf-8").split()
									if vp == vn:
										tmp = "	".join((vp, np, nn,))
										fwrt.write(tmp.encode("utf-8"))
										fwrt.write("\n")
									else:
										print("Invalid data")

if __name__ == "__main__":
	handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"))
