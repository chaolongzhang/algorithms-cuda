CC=nvcc

# Platform: WIN(Windows)/Linux/MacOS
Platform=WIN
OPENCV_Include="C:\opencv\build\include"
OPENCV_LIB_DIR="C:\opencv\build\x64\vc14\lib"
OPENCV_LIBS = "-lopencv_world310d"
CFLAGS=-I. -I../cuda -I../cpp -I$(OPENCV_Include)
LINK=-L$(OPENCV_LIB_DIR)  $(OPENCV_LIBS)
STD_VERSION=--std=c++11

CPP_FILES=helper.cpp test.cpp timeclock.cpp ../cuda/RGF_cuda.cu
CUDA_FILES=helper.cpp test.cu timeclock.cpp ../cuda/RGF_cuda.cu

all: $(CPP_FILES) $(CUDA_FILES)
	nvcc -o RGF_cpp.exe $(CPP_FILES) $(STD_VERSION) $(CFLAGS) $(LINK)
	nvcc -o RGF_cuda.exe $(CUDA_FILES) $(STD_VERSION) $(CFLAGS) $(LINK) -DGPU

cpp: $(CPP_FILES)
	nvcc -o RGF_cpp.exe $(CPP_FILES) $(STD_VERSION) $(CFLAGS) $(LINK)

cuda: $(CUDA_FILES)
    nvcc -o RGF_cuda.exe $(CUDA_FILES) $(STD_VERSION) $(CFLAGS) $(LINK) -DGPU

clean:
    rm *.exe *.exp *.lib