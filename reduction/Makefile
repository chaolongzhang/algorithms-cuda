CC=nvcc
CFLAGS=-I /Developer/NVIDIA/CUDA-7.5/samples/7_CUDALibraries -I /Developer/NVIDIA/CUDA-7.5/samples/common/inc

all: reduction_thrust.cu reduction_cuda.cu
	$(CC) -o reduction_cuda.exe reduction_cuda.cu  $(CFLAGS)
	$(CC) -o reduction_thrust.exe reduction_thrust.cu $(CFLAGS) 

cuda: reduction_cuda.cu
	$(CC) -o reduction_cuda.exe reduction_cuda.cu  $(CFLAGS)

thrust: reduction_thrust.cu
	$(CC) -o reduction_thrust.exe reduction_thrust.cu $(CFLAGS) 


clean:
	rm *.exe