
# ponizsza linijka przyda sie dla cuda 3.0 i gcc 4.4, bo sei gryza
BUG_FIX = -Xcompiler -D__builtin_stdarg_start=__builtin_va_start

FLAGS = $(BUG_FIX) -Xcompiler -fpic
#FLAGS = --device-emulation --ptxas-options=-v -maxrregcount=64 -g -G
SWIG_C_FILES_DIR = ../swig/c
CUDA_LIB = -L/usr/local/cuda/lib -lcudart 



JAVA = -I$${JAVA_HOME}/include -I$${JAVA_HOME}/include/linux



install: test lib

#to jest biblioteka CUDA (dla JNI)
lib: rozgrywka.o 
	nvcc -shared $(CUDA_LIB) $(FLAGS) $(JAVA) $(SWIG_C_FILES_DIR)/texas_swig.c $(SWIG_C_FILES_DIR)/generator_liczb_losowych.c  $(SWIG_C_FILES_DIR)/ai_texas_swig.c rozgrywka.o  -o libtests_texas.so

#to jest standalone program C++ do obliczania osobnikow
test:
	nvcc funkcje/test_obj_function.cu $(CUDA_LIB) $(FLAGS) $(JAVA) -o test_obj_fun

#tu jest glowny kernel CUDA
rozgrywka.o:
	nvcc -c $(CUDA_LIB) $(JAVA) $(FLAGS) funkcje/rozgrywka.cu
