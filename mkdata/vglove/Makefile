CC = gcc
#For older gcc, use -O3 or -O2 instead of -Ofast
CFLAGS = -lm -pthread -Ofast -march=native -funroll-loops -Wno-unused-result

all: vglove

vglove : vg.c
	$(CC) vg.c -o vg $(CFLAGS)

clean:
	rm -rf vg