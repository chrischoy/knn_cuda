# Includes
PYTHON_INCLUDE = /home/chrischoy/anaconda/include/python2.7

# CUDA compilation rules
NVCC = nvcc
NVCCFLAGS = -m64 -Xcompiler '-fPIC'
CUDA_INCLUDES = -I$(PYTHON_INCLUDE)
CUDA_LIBS = -lcuda -lcudart -lcudadevrt -L/usr/local/cuda-7.0/lib64
CUDA_TARGET = knn_cuda

# Library compilation rules
CC = g++
CCFLAGS = -fPIC
LDFLAGS = -shared
INCLUDES = -I$(PYTHON_INCLUDE)
LIBS = -lboost_python -lpython2.7 $(CUDA_LIBS)
TARGET = knn

all: clean $(TARGET).so
 
$(TARGET).so: $(TARGET).o $(CUDA_TARGET).o $(CUDA_TARGET)_link.o
	$(CC) $(LDFLAGS) $? $(LIBS) -o $@

$(TARGET).o: $(TARGET).cpp
	$(CC) $(INCLUDES) $(CCFLAGS) -c $?

$(CUDA_TARGET).o: $(CUDA_TARGET).cu
	$(NVCC) $(NVCCFLAGS) -dc $?

$(CUDA_TARGET)_link.o: $(CUDA_TARGET).o
	$(NVCC) $(NVCCFLAGS) $(CUDA_LIBS) -dlink -o $@ $?

clean:
	rm -f $(TARGET).so $(CUDA_TARGET).o $(TARGET).o $(CUDA_TARGET)_link.o
