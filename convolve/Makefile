
CC=nvcc
CFLAGS=-I /Developer/NVIDIA/CUDA-7.5/samples/7_CUDALibraries -I /Developer/NVIDIA/CUDA-7.5/samples/common/inc

all: convolve.cu convolvefft.cu 
	$(CC) -o convolve.exe convolve.cu  $(CFLAGS) 
	$(CC) -o convolvefft.exe convolvefft.cu  $(CFLAGS) -lcufft

convolve: convolve.cu 
	$(CC) -o convolve.exe convolve.cu  $(CFLAGS)

convolvefft: convolvefft.cu 
	$(CC) -o convolvefft.exe convolvefft.cu  $(CFLAGS) -lcufft


clean:
	rm *.exe