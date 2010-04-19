
FLAGS = --device-emulation
#FLAGS = --device-emulation --ptxas-options=-v -maxrregcount=64 -g -G
SWIG_C_FILES_DIR = ../swig/c
CUDA_LIB = -L/usr/local/cuda/lib -lcudart 

JAVA = -I$${JAVA_HOME}/include -I$${JAVA_HOME}/include/linux



install: lib test

lib: rozgrywka.o 
	nvcc -shared $(CUDA_LIB) $(FLAGS) $(JAVA) rozgrywka.o $(SWIG_C_FILES_DIR)/texas_swig.c $(SWIG_C_FILES_DIR)/ai_texas_swig.c -o libtests_texas.so
#lib: rozgrywka.o 
#	nvcc -shared $(CUDA_LIB) $(FLAGS) $(JAVA) osobnik.o -o libtests_texas.so

test:
	gcc funkcje/test_obj_function.h rozgrywka.o $(CUDA_LIB) $(JAVA) $(SWIG_C_FILES_DIR)/texas_swig.c $(SWIG_C_FILES_DIR)/ai_texas_swig.c -o test_cuda_obj_fun

rozgrywka.o:
	nvcc -c $(FLAGS) funkcje/rozgrywka.cu

