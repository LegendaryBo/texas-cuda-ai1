#include "../struktury/gracz.h"
#include "../struktury/gra.h"
#include "../funkcje/rozgrywka.h"

#include "../funkcje/rezultaty.h"
#include "../funkcje/osobnik.h"
#include "../funkcje/cuda_zlecenia.h"

#include "../struktury/ile_grac_r1.h"
#include "../struktury/ile_grac_rx.h"
#include "../struktury/dobijanie_r1.h"
#include "../struktury/dobijanie_rx.h"
#include "../struktury/stawka_r1.h"
#include "../struktury/stawka_rx.h"
#include "../struktury/czy_grac_r1.h"
#include "../struktury/czy_grac_rx.h"
#include "../struktury/reguly.h"
#include "../struktury/zlecenie.h"


#include "../funkcje/reguly_ilegrac.h"
#include "../funkcje/reguly_stawka.h"
#include "../funkcje/reguly_dobijania.h"
#include "../funkcje/reguly_czygrac.h"
#include "../funkcje/reguly.h"

#include <stdio.h>
#include <stdlib.h>

int rozmiar_genomu = 1234;
//float minimal_bid = 10.0;



void getBilans(Gra *gra,  float *output ) {
	for (int i=0; i < 6; i++)
		output[i] = gra->gracze[i].bilans;
};


Gra *getGraPTR() {
	Gra *gra = new Gra;
	return gra;
};



int getJakisHashcode(int *osobnik, int dimension) {

	int wynik=13;
	for (int i=0; i < dimension; i++) {
		if (getBitHOST(osobnik, i)  == 1)
			wynik = wynik*183 +191;
		else
			wynik = wynik*521 + 31;
	}
	return wynik;

}


Zlecenie **stworzZlecenia(int ktory_nasz, int *osobniki, int N) {

	int licznik=0;

	Zlecenie **zlecenia = (Zlecenie**) malloc(sizeof(Zlecenie*) * N);
	for (int i=0; i < N; i++) {
		zlecenia[i] = noweZlecenie( ktory_nasz,
								  (licznik+1) %100,
								  (licznik+2) %100,
								  (licznik+3) %100,
								  (licznik+4) %100,
								  (licznik+5) %100,
								ktory_nasz, i, osobniki);
		licznik+=6;
	}


	return zlecenia;

}







int **getIndividualPTRPTR(int size) {

	int **wskaznik = (int**)malloc( sizeof(int*) * size );


	return wskaznik;
}

void setIndividualPTR(int *osobnik, int **partnerzy, int index) {
	partnerzy[index] = osobnik;
}


float obliczZlecenia(Zlecenie **zlecenie, int ile_zlecen, int ile_intow) {

	float suma=0.0;

	for (int i=0; i < ile_zlecen; i++) {

		Reguly *regula = getReguly();
		Gra *gra = new Gra;

		nowaGra(   &zlecenie[i]->osobniki[0] + zlecenie[i]->indexOsobnika[0] *  ile_intow ,
				&zlecenie[i]->osobniki[0] + zlecenie[i]->indexOsobnika[1] *  ile_intow,
				&zlecenie[i]->osobniki[0] + zlecenie[i]->indexOsobnika[2] *  ile_intow,
				&zlecenie[i]->osobniki[0] + zlecenie[i]->indexOsobnika[3] *  ile_intow,
				&zlecenie[i]->osobniki[0] + zlecenie[i]->indexOsobnika[4] *  ile_intow ,
				&zlecenie[i]->osobniki[0] + zlecenie[i]->indexOsobnika[5] *  ile_intow ,
				zlecenie[i]->nrRozdania, 3, gra);

		zlecenie[i]->wynik =rozegrajPartieHOST(gra, 0, regula);
		suma+=zlecenie[i]->wynik;
//		printf("gra numer %d - ", i);
//		printf(" %f \n", suma/ile_zlecen);

		free(gra);
	}
	return suma/ile_zlecen;
}


void rozegrajNGier(int ktory_nasz, int **osobniki, float *wynik, int N, int liczba_intow) {

	wynik[0]=0.0;


	int *osobniki_statyczna_tablica = (int*) malloc(sizeof(int) * 101 * liczba_intow);

	for (int i=0; i < 101; i++) {
		for (int j=0; j < liczba_intow; j++)
			osobniki_statyczna_tablica[ j + i * liczba_intow  ] = *(&(osobniki[i])[j]);
	}

	Zlecenie **zlecenia = stworzZlecenia(ktory_nasz, osobniki_statyczna_tablica, N);

	wynik[0] = obliczZlecenia(zlecenia, N, liczba_intow);



	for (int i=0; i < N; i++) {
		free(zlecenia[i]);
	}
	free(zlecenia);

}


//64kb pamieci stalej
static __constant__ int osobniki_const[256*64];

__global__ void obliczZlecenie(int liczbaGier, Zlecenie *zlecenia_cuda, float *wyniki_device, int ile_intow, Gra *gra, Reguly *reguly) {

	//extern __shared__ int temp_count[];

	register int nr_zlecenia;
	nr_zlecenia = blockIdx.x*blockDim.x + threadIdx.x;



	if (nr_zlecenia>=liczbaGier)
	  return;

	(zlecenia_cuda + nr_zlecenia ) -> osobniki = &osobniki_const[0];
	(zlecenia_cuda + nr_zlecenia ) -> nrRozdania = nr_zlecenia;

	(zlecenia_cuda + nr_zlecenia ) -> indexOsobnika[0] = 100;
	(zlecenia_cuda + nr_zlecenia ) -> indexOsobnika[1] = (nr_zlecenia*6 + 1)%100;
	(zlecenia_cuda + nr_zlecenia ) -> indexOsobnika[2] = (nr_zlecenia*6 + 2)%100;
	(zlecenia_cuda + nr_zlecenia ) -> indexOsobnika[3] = (nr_zlecenia*6 + 3)%100;
	(zlecenia_cuda + nr_zlecenia ) -> indexOsobnika[4] = (nr_zlecenia*6 + 4)%100;
	(zlecenia_cuda + nr_zlecenia ) -> indexOsobnika[5] = (nr_zlecenia*6 + 5)%100;

	(zlecenia_cuda + nr_zlecenia ) -> indexGracza = 100;
//	printf("indeks pierwszego osobnika %d \n",(nr_zlecenia*6 + 1)%100);

	//__shared__ Gra gra[4];

	nowaGra(   &osobniki_const[0] + 100 *  ile_intow ,
			&osobniki_const[0] + (nr_zlecenia*6 + 1)%100 *  ile_intow,
			&osobniki_const[0] + (nr_zlecenia*6 + 2)%100 *  ile_intow,
			&osobniki_const[0] + (nr_zlecenia*6 + 3)%100 *  ile_intow,
			&osobniki_const[0] + (nr_zlecenia*6 + 4)%100 *  ile_intow,
			&osobniki_const[0] + (nr_zlecenia*6 + 5)%100 *  ile_intow,
			nr_zlecenia, 3, &gra[nr_zlecenia]);

	float bla = rozegrajPartieDEVICE(&gra[nr_zlecenia], 0, reguly);

//	int spasowani[6];
//	int wygrani[6];
//	for (int i=0; i < 6; i++)
//		spasowani[i]=0;
//	int ile = wygrany(& ( gra[nr_zlecenia].rozdanie ),
//			&spasowani[0],
//			&wygrani[0]);
//
//	float bla=rozegrajPartie(&gra[nr_zlecenia], 0, reguly);
//
//	for  (int i=0; i < ile; i++) {
//		bla += wygrani[i] * wygrani[i];
//	}


//	printf("wynik to %f \n", bla);
	//wyniki_device[ nr_zlecenia ] = bla;

	wyniki_device[ nr_zlecenia ] = bla;


}

// metoda sprawdza, czy wystapil w GPU jakis blad
void obsluzBlad(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) // sprawdzamy, czy blad wystapil
    {
        fprintf(stderr, "Blad Cuda : %s: %s.\n", msg,
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}




void rozegrajNGierCUDA(int ktory_nasz, int **osobniki, float *wynik, int N,
int liczba_intow, int block_size) {


	putenv("CUDA_PROFILE=1");
	putenv("CUDA_PROFILE_LOG=/home/kacper/cudaProfiler/profiler_log");

	putenv("CUDA_PROFILE_CONFIG=/home/kacper/cudaProfiler/cuda_profiler.cfg");

	int *osobniki_statyczna_tablica = (int*) malloc(sizeof(int) * 101 * liczba_intow);

	for (int i=0; i < 101; i++) {
		for (int j=0; j < liczba_intow; j++)
			osobniki_statyczna_tablica[ j + i * liczba_intow  ] = *(&(osobniki[i])[j]);
	}

	//printf("liczba watkow na grid %d\n", block_size);
	//printf("liczba partii %d\n", N);
	//printf("wielkosc osobnika %d\n", liczba_intow*4);

	Gra *gry_cuda;
	size_t size_gry = sizeof(Gra)*N;
	cudaMalloc((void **) &gry_cuda, size_gry);

	Reguly *reguly_cuda;
	Reguly *reguly_host = getReguly();
	size_t size_reguly = sizeof(Reguly);
	cudaMalloc((void **) &reguly_cuda, size_reguly);



	float *wyniki_device;
	size_t size_wyniki = sizeof(float)*N;
	cudaMalloc((void **) &wyniki_device, size_wyniki);
	float *wyniki_host;
	wyniki_host = (float *)malloc( size_wyniki );

	Zlecenie *zlecenia_cuda;
	size_t size_zlecenie = sizeof(Zlecenie)*N;
	cudaMalloc((void **) &zlecenia_cuda, size_zlecenie);

    	//int sharedMemSize = block_size * sizeof(int) * 10;

        cudaMemcpyToSymbol(osobniki_const, osobniki_statyczna_tablica, liczba_intow*sizeof(int)*101);
	obsluzBlad("kopiowanie osobnikow na karte");
	//cudaMemcpy(osobniki_cuda, osobniki_statyczna_tablica, liczba_intow*sizeof(int)*101, cudaMemcpyHostToDevice);
	cudaMemcpy(reguly_cuda, reguly_host, sizeof(Reguly), cudaMemcpyHostToDevice);

	int nBlocks = N/block_size + 1;
	if (N%block_size==0)
	  nBlocks--;

	obsluzBlad("kopiowanie pozostalych danych na karte");
	//obliczZlecenie <<< nBlocks, block_size,  sharedMemSize>>> (N, zlecenia_cuda, wyniki_device, liczba_intow, gry_cuda, reguly_cuda);
	cudaThreadSynchronize();
	obsluzBlad("uruchomienia kernela");

	cudaMemcpy(wyniki_host, wyniki_device, size_wyniki  , cudaMemcpyDeviceToHost);

        obsluzBlad("kopiowanie wynikow z karty");

	float suma = 0.0;
	for (int i=0; i < N; i++) {
		suma += wyniki_host[i];
		//printf("\nCUDA bilans po grze nr %d",sizeof() );
		//printf("\n wskaznik: %d", wyniki_host[i]);
		printf("ma wynik %f ", (suma/N) );
	}

	wynik[0] = suma/N;
}








void destruktorGra(Gra *gra) {
	free(gra);
};

void destruktorInt(int *ptr) {
	free(ptr);
};

void destruktorKodGraya(KodGraya *kodGraya) {
	free(kodGraya);
};

extern void destruktorRozdanie(Rozdanie *rozdanie) {
	free(rozdanie);
};

extern void destruktorHand(Hand *hand) {
	free(hand);
};
