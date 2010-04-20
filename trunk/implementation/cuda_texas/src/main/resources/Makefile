
FLAGS = --device-emulation
#FLAGS = --device-emulation --ptxas-options=-v -maxrregcount=64 -g -G
SWIG_C_FILES_DIR = ../swig/c
CUDA_LIB = -L/usr/local/cuda/lib -lcudart 

JAVA = -I$${JAVA_HOME}/include -I$${JAVA_HOME}/include/linux



install: lib test

#to jest biblioteka CUDA (dla JNI)
lib: rozgrywka.o 
	nvcc -shared $(CUDA_LIB) $(FLAGS) $(JAVA) $(SWIG_C_FILES_DIR)/texas_swig.c $(SWIG_C_FILES_DIR)/ai_texas_swig.c -o libtests_texas.so

#to jest standalone program C++ do obliczania osobnikow
test: rozgrywka.o
	nvcc funkcje/test_obj_function.cu rozgrywka.o $(CUDA_LIB) $(JAVA) $(SWIG_C_FILES_DIR)/texas_swig.c $(SWIG_C_FILES_DIR)/ai_texas_swig.c -o test_obj_fun

#tu jest glowny kernel CUDA
rozgrywka.o:
	nvcc -c $(FLAGS) funkcje/rozgrywka.cu

