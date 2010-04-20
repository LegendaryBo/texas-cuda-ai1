#include "rozgrywka.cu"
#include "osobnikiIO.h"

#include <stdio.h>
#include <stdlib.h>

int liczba_intow = 73;
int liczba_gier=5;
int LICZBA_OSOBNIKOW=101;
int seed=465;
int a=243812;
int b=353542;
int m=6592981;


int nextInt(int modulo) {
	seed = (a*seed +b) % m;
	return seed%modulo;
}

int nextInt() {
	seed = (a*seed +b) % m;
	return seed;
}

float obliczFunkcjeCelu(int **osobniki){
	float *wynik = (float*)malloc(sizeof(float));
	rozegrajNGierCUDA(101,
			osobniki, wynik,
			liczba_gier, //liczba gier
			liczba_intow,
			16 );
	for (int i=0; i < liczba_gier; i++) {
		printf("%f\n",wynik[0]);
	}
	return wynik[0];
}

int main() {

	int *osobnik_obliczany = (int*)malloc( sizeof(int) * LICZBA_OSOBNIKOW );
	for (int i=0; i < liczba_intow; i++)
		osobnik_obliczany[i] = nextInt();

	// ladowanie osobnikow
	ZbiorOsobnikow *zbior_osobnikow = odczytajOsobnikiZKatalogu(11,
			"home/kacper/workspace/svn/texas-cuda-ai1/implementation/cuda_texas/target/classes/texas_individuale/",
			liczba_intow);
	int liczbaOsobnikow = zbior_osobnikow->liczba_osobnikow;


	int **osobniki =  (int**)malloc( sizeof(int*) * LICZBA_OSOBNIKOW );
	for (int i=0; i < 100; i++) {
		int wylosowany = nextInt(liczbaOsobnikow);
		osobniki[i] = zbior_osobnikow->osobniki[ wylosowany ]->geny;
	}
	osobniki[100] = osobnik_obliczany;
	obliczFunkcjeCelu(osobniki);
}








