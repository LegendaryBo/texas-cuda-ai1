
FLAGS = --device-emulation
#FLAGS = --device-emulation --ptxas-options=-v -maxrregcount=64 -g -G
SWIG_C_FILES_DIR = ../../../target/swig/c
CUDA_LIB = -L/usr/local/cuda/lib -lcudart 

JAVA = -I$${JAVA_HOME}/include -I$${JAVA_HOME}/include/linux



install: lib

lib: rozgrywka.o 
	nvcc -shared $(CUDA_LIB) $(FLAGS) $(JAVA) rozgrywka.o $(SWIG_C_FILES_DIR)/texas_swig.c $(SWIG_C_FILES_DIR)/ai_texas_swig.c -o libtests_texas.so
#lib: rozgrywka.o 
#	nvcc -shared $(CUDA_LIB) $(FLAGS) $(JAVA) osobnik.o -o libtests_texas.so

rozgrywka.o:
	nvcc -c $(FLAGS) c/funkcje/rozgrywka.cu

