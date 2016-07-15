CC=nvcc
CFLAGS=-I /Developer/NVIDIA/CUDA-7.5/samples/7_CUDALibraries -I /Developer/NVIDIA/CUDA-7.5/samples/common/inc

all: scan.cu
	$(CC) -o scan.exe scan.cu $(CFLAGS)
	
clean:
	rm *.exe