#encoding: utf-8

import sys
from random import shuffle

def handle(fsrc, frs):
	content = []
	with open(fsrc) as f:
		for line in f:
			tmp = line.strip()
			if tmp:
				content.append(tmp.decode("utf-8"))
	shuffle(content)
	with open(frs, "w") as f:
		content.append("")
		tmp = "\n".join(content)
		f.write(tmp.encode("utf-8"))

if __name__ == "__main__":
	handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"))
