#include "rozgrywka.cu"
#include "osobnikiIO.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>

int liczba_intow = 73;
int liczba_genow=2329;
int LICZBA_OSOBNIKOW=101;
int seed=465;
int a=65537;
int b=257;

//int m=274177;


int nextInt(int modulo) {

	seed = a*seed + b;
	if (seed < 0)
		seed -=seed;
	return seed%modulo;
}

int nextInt() {
	seed =  a*seed + b;
	return seed;
}

float obliczFunkcjeCelu(int **osobniki, int liczba_gier, int liczba_watkow){
	float *wynik = (float*)malloc(sizeof(float));
	rozegrajNGierCUDA(101,
			osobniki, wynik,
			liczba_gier, //liczba gier
			liczba_intow,
			16,
			100);

//	for (int i=0; i < liczba_gier; i++) {
//		printf("%f\n",wynik[0]);
//	}
	return wynik[0];
}

int getBit(int *osobnik, int ktory_bit) {
	return ( (unsigned int) (osobnik[ktory_bit / 32] & (1 << ktory_bit % 32)) )>> (ktory_bit
			% 32);
};

void wypiszOsobnika(int *osobnik) {

	int suma = 0;
//	for (int i=0; i < 2329; i++) {
//		suma += osobnik[i];
////		for (int j=0; j < 2329; j++)
//		printf("%d",getBit(osobnik, i));
//	}
	for (int i=0; i < liczba_intow; i++) {
		suma += osobnik[i];
	}
//	printf("%d",suma);
//	printf("\n");
}

int obliczHashOsobnika(int *osobnik, int dlugoscOsobnikaWIntach) {
	int hash=0;
	for (int i=0; i < dlugoscOsobnikaWIntach; i++) {
		hash+=i*osobnik[i];
	}
	return hash;
}

int main( int argc, char* argv[] ) {

	seed=465; // resetujemy ustawienia generatora

	if (argc < 3) {
		printf("Blad, wymagane sa 3 parametry:\n");
		printf("1) bezwzbledna sciezka do katalogu z osobnikami:\n");
		printf("2) liczba gier na jedna funkcje celu:\n");
		printf("3) liczba watkow na jednego osobnika:\n");
		printf("4) liczba obliczanych osobnikow:\n");
	}


	// ladowanie osobnikow
	ZbiorOsobnikow *zbior_osobnikow = odczytajOsobnikiZKatalogu(11,
			argv[1],
//			"/home/railman/workspace/svn/texas-cuda-ai1/implementation/cuda_texas/target/classes/texas_individuale/",
			liczba_intow);
	int liczbaOsobnikow = zbior_osobnikow->liczba_osobnikow;

	int liczba_gier = atoi(argv[2]);
	int liczba_watkow = atoi(argv[3]);
	int liczba_osobnikow = atoi(argv[4]);

	printf("Obliczanie funkcji celu na GPU \n");
	printf("sumaryczna liczba osobnikow wczytanych z pliku %d \n", liczbaOsobnikow);
	printf("liczba gier: %d \n", liczba_gier);
	printf("liczba watkow: %d \n", liczba_watkow);
	printf("liczba obliczanych osobnikow: %d \n", liczba_osobnikow);

	int **osobniki =  (int**)malloc( sizeof(int*) * LICZBA_OSOBNIKOW );
	for (int i=0; i < 100; i++) {
		int wylosowany = nextInt(liczbaOsobnikow);
		osobniki[i] = zbior_osobnikow->osobniki[ wylosowany ]->geny;
//		printf("\n osobnik nr %d \n",(i+1));
		wypiszOsobnika(osobniki[i]);
	}

	int *osobnik_obliczany = (int*)malloc( sizeof(int) * LICZBA_OSOBNIKOW );
	for (int k=0; k < liczba_osobnikow; k++) {

		int losowa = nextInt(liczbaOsobnikow);
		osobniki[100] = zbior_osobnikow->osobniki[ losowa ]->geny;
		float wynik = obliczFunkcjeCelu(osobniki, liczba_gier, liczba_watkow);
		printf("osobnik nr %d - ", (k+1));
//		printf(" hash osobnika %d", obliczHashOsobnika(osobniki[100], liczba_intow)  );
		printf(" %f \n",wynik);
	}
}








