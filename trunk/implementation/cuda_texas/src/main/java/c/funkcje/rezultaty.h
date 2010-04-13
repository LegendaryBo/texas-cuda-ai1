#include "../struktury/texas_struktury.h"
#include "../struktury/hand.h"
#include "../struktury/rozdanie.h"

#ifndef REZULTATY_H
#define REZULTATY_H

// kazdemu watkawi dajemy 10 bajtow
#define SHARED_MEM_THREAD_SIZE 10

extern "C" {

	extern __shared__ int fast_var[];

	__host__ void iloscKartTejSamejWysokosciHOST(Hand *rezultat, int *wynik) {
		
		int temp_count[15];

		int i;

		for (i = 0; i < 15; i++)
			temp_count[i] = 0;

		for (i = 0; i < rezultat->ile_kart; i++)
			temp_count[rezultat->karty[i].wysokosc]++;

		//__shared__ int first_place;
		int first_place=0;

		//__shared__ int second_place;
		int second_place=0;


		for (i = 1; i < 15; i++) {
			if (temp_count[i] < first_place && temp_count[i] > second_place)
				second_place = temp_count[i];

			if (temp_count[i] >= first_place) {
				second_place = first_place;
				first_place = temp_count[i];
			}
		}

		wynik[0] = first_place;
		wynik[1] = second_place;
	};
	__device__ void iloscKartTejSamejWysokosciDEVICE(Hand *rezultat, int *wynik) {
		
		char *temp_count = (char*)&fast_var[ threadIdx.x*SHARED_MEM_THREAD_SIZE ];	// zmienna w pamieci dzielonej
		//int temp_count[15];

		int i;

		for (i = 0; i < 15; i++)
			temp_count[i] = 0;

		for (i = 0; i < rezultat->ile_kart; i++)
			temp_count[rezultat->karty[i].wysokosc]++;

		//__shared__ int first_place;
		int first_place=0;

		//__shared__ int second_place;
		int second_place=0;


		for (i = 1; i < 15; i++) {
			if (temp_count[i] < first_place && temp_count[i] > second_place)
				second_place = temp_count[i];

			if (temp_count[i] >= first_place) {
				second_place = first_place;
				first_place = temp_count[i];
			}
		}

		wynik[0] = first_place;
		wynik[1] = second_place;
	};



	__host__ int jestStreetHOST(Hand *reka) {

		int temp_count[15];	
		//__shared__ int temp_count[15 * 16];
		int offset=0;
		//offset = 0;
		//offset = threadIdx.x*15;
		int i;
		//int count[15];
		for (i = 0; i < 15; i++)
			temp_count[offset +  i] = 0;

		for (i = 0; i < reka->ile_kart; i++)
			temp_count[offset +  reka->karty[i].wysokosc]++;

		temp_count[offset +  1] = temp_count[offset +  14]; // bo as sluzy tez za jedynke

		int dno_streeta;
                dno_streeta=1;
		int szczyt_streeta;
                szczyt_streeta=1;
		int jest_street;
                jest_street=0;

		for (i = 1; i < 15; i++) {

			if (temp_count[offset +  i] > 0)
				szczyt_streeta = i;
			else
				dno_streeta = (int) (i + 1);

			if (szczyt_streeta - dno_streeta >= 4) {
				jest_street = 1;
			}
		}

		return jest_street;
	}
	;
	__device__ int jestStreetDEVICE(Hand *reka) {

		int temp_count[15];	
		//__shared__ int temp_count[15 * 16];
		int offset=0;
		//offset = 0;
		//offset = threadIdx.x*15;
		int i;
		//int count[15];
		for (i = 0; i < 15; i++)
			temp_count[offset +  i] = 0;

		for (i = 0; i < reka->ile_kart; i++)
			temp_count[offset +  reka->karty[i].wysokosc]++;

		temp_count[offset +  1] = temp_count[offset +  14]; // bo as sluzy tez za jedynke

		int dno_streeta;
                dno_streeta=1;
		int szczyt_streeta;
                szczyt_streeta=1;
		int jest_street;
                jest_street=0;

		for (i = 1; i < 15; i++) {

			if (temp_count[offset +  i] > 0)
				szczyt_streeta = i;
			else
				dno_streeta = (int) (i + 1);

			if (szczyt_streeta - dno_streeta >= 4) {
				jest_street = 1;
			}
		}

		return jest_street;
	}
	;


	__host__ int jestKolorHOST(Hand *reka) {

		int count[15];

		int i;

		for (i = 0; i < 5; i++)
			count[i] = 0;

		for (i = 0; i < reka->ile_kart; i++) {
			count[reka->karty[i].kolor]++;
		}

		if (count[1] >= 5 || count[2] >= 5 || count[3] >= 5 || count[4] >= 5)
			return 1;

		return 0;
	};
	__device__ int jestKolorDEVICE(Hand *reka) {

		int count[15];

		int i;

		for (i = 0; i < 5; i++)
			count[i] = 0;

		for (i = 0; i < reka->ile_kart; i++) {
			count[reka->karty[i].kolor]++;
		}

		if (count[1] >= 5 || count[2] >= 5 || count[3] >= 5 || count[4] >= 5)
			return 1;

		return 0;
	};
	
	

	__host__ int jestPokerHOST(Hand *reka) {

		int count[15][5];
		int i;
		int j;
		for (i = 0; i < 15; i++) {
			for (j = 0; j < 5; j++) {
				count[i][j] = 0;

			}
		}

		for (i = 0; i < reka->ile_kart; i++) {
			count[reka->karty[i].wysokosc][reka->karty[i].kolor]++;
		}
		count[1][1] = count[14][1];
		count[1][2] = count[14][2];
		count[1][3] = count[14][3];
		count[1][4] = count[14][4];

		int dno_pokera;
		dno_pokera = 1;
		int szczyt_pokera;
		szczyt_pokera = 1;

		for (i = 1; i < 15; i++) {

			if (count[i][1] > 0)
				szczyt_pokera = i;
			else
				dno_pokera = i + 1;

			if (szczyt_pokera - dno_pokera >= 4) {
				return 1;
			}
		}

		dno_pokera = 1;
		szczyt_pokera = 1;

		for (i = 1; i < 15; i++) {

			if (count[i][2] > 0)
				szczyt_pokera = i;
			else
				dno_pokera = i + 1;

			if (szczyt_pokera - dno_pokera >= 4) {
				return 1;
			}
		}

		dno_pokera = 1;
		szczyt_pokera = 1;

		for (int i = 1; i < 15; i++) {

			if (count[i][3] > 0)
				szczyt_pokera = i;
			else
				dno_pokera = i + 1;

			if (szczyt_pokera - dno_pokera >= 4) {
				return 1;
			}
		}

		dno_pokera = 1;
		szczyt_pokera = 1;

		for (int i = 1; i < 15; i++) {

			if (count[i][4] > 0)
				szczyt_pokera = i;
			else
				dno_pokera = i + 1;

			if (szczyt_pokera - dno_pokera >= 4) {
				return 1;
			}
		}
		return 0;

	};
	__device__ int jestPokerDEVICE(Hand *reka) {

		int count[15][5];
		int i;
		int j;
		for (i = 0; i < 15; i++) {
			for (j = 0; j < 5; j++) {
				count[i][j] = 0;

			}
		}

		for (i = 0; i < reka->ile_kart; i++) {
			count[reka->karty[i].wysokosc][reka->karty[i].kolor]++;
		}
		count[1][1] = count[14][1];
		count[1][2] = count[14][2];
		count[1][3] = count[14][3];
		count[1][4] = count[14][4];

		int dno_pokera;
		dno_pokera = 1;
		int szczyt_pokera;
		szczyt_pokera = 1;

		for (i = 1; i < 15; i++) {

			if (count[i][1] > 0)
				szczyt_pokera = i;
			else
				dno_pokera = i + 1;

			if (szczyt_pokera - dno_pokera >= 4) {
				return 1;
			}
		}

		dno_pokera = 1;
		szczyt_pokera = 1;

		for (i = 1; i < 15; i++) {

			if (count[i][2] > 0)
				szczyt_pokera = i;
			else
				dno_pokera = i + 1;

			if (szczyt_pokera - dno_pokera >= 4) {
				return 1;
			}
		}

		dno_pokera = 1;
		szczyt_pokera = 1;

		for (int i = 1; i < 15; i++) {

			if (count[i][3] > 0)
				szczyt_pokera = i;
			else
				dno_pokera = i + 1;

			if (szczyt_pokera - dno_pokera >= 4) {
				return 1;
			}
		}

		dno_pokera = 1;
		szczyt_pokera = 1;

		for (int i = 1; i < 15; i++) {

			if (count[i][4] > 0)
				szczyt_pokera = i;
			else
				dno_pokera = i + 1;

			if (szczyt_pokera - dno_pokera >= 4) {
				return 1;
			}
		}
		return 0;

	};	
	
	

	__device__ __host__  void buildPoker(Hand *reka) {

		reka->rezultat.reszta2=0;
		reka->rezultat.reszta3=0;
		reka->rezultat.reszta4=0;
		reka->rezultat.reszta5=0;
		reka->rezultat.poziom=0;

		int count[5];
		for (int i = 0; i < 5; i++)
			count[i] = 0;

		for (int i = 0; i < reka->ile_kart; i++)
			count[reka->karty[i].kolor]++;

		int max_kolor = 0;
		for (int i = 1; i <= 4; i++) {
			if (count[i] >= 5)
				max_kolor = i;
		}

		for (int i = 0; i < reka->ile_kart; i++) {
			if (reka->karty[i].wysokosc > reka->rezultat.reszta1
					&& reka->karty[i].kolor == max_kolor)
				reka->rezultat.reszta1 = reka->karty[i].wysokosc;
		}

		reka->rezultat.poziom = 9;

	};

	__device__ __host__  void buildKareta(Hand *reka) {

		reka->rezultat.reszta2=0;
		reka->rezultat.reszta3=0;
		reka->rezultat.reszta4=0;
		reka->rezultat.reszta5=0;

		int count[15];
		for (int i = 0; i < 15; i++)
			count[i] = 0;

		for (int i = 0; i < reka->ile_kart; i++) {
			count[reka->karty[i].wysokosc]++;
		}

		for (int i = 14; i >= 2; i--) {

			if (count[i] == 4)
				reka->rezultat.reszta1 = i;

			if (count[i] > 0 && count[i] < 4 && i > reka->rezultat.reszta2)
				reka->rezultat.reszta2 = i;
		}

		reka->rezultat.poziom = 8;
	};


	__device__ __host__  void buildFull(Hand *reka) {


		reka->rezultat.reszta2=0;
		reka->rezultat.reszta3=0;
		reka->rezultat.reszta4=0;
		reka->rezultat.reszta5=0;

		int count[15];
		for (int i = 0; i < 15; i++)
			count[i] = 0;

		for (int i = 0; i < reka->ile_kart; i++) {
			count[reka->karty[i].wysokosc]++;
		}

		for (int i = 14; i >= 2; i--) {

			if (count[i] == 3)
				reka->rezultat.reszta1 = i;

			if (count[i] == 2 && i > reka->rezultat.reszta2) {
				reka->rezultat.reszta2 = i;

			}
		}

		reka->rezultat.poziom = 7;

	};

	__device__ __host__  void buildKolor(Hand *reka) {

		int count[5];
		for (int i = 0; i < 5; i++)
			count[i] = 0;

		for (int i = 0; i < reka->ile_kart; i++) {
			count[reka->karty[i].kolor]++;
		}

		int max_kolor = 0;
		for (int i = 1; i <= 4; i++) {
			if (count[i] >= 5)
				max_kolor = i;
		}

		reka->rezultat.reszta1=0;
		reka->rezultat.reszta2=0;
		reka->rezultat.reszta3=0;
		reka->rezultat.reszta4=0;
		reka->rezultat.reszta5=0;

		for (int i = 0; i < reka->ile_kart; i++) {
			if (reka->karty[i].wysokosc > reka->rezultat.reszta1
					&& reka->karty[i].kolor == max_kolor)
				reka->rezultat.reszta1 = reka->karty[i].wysokosc;
		}


		reka->rezultat.poziom = 6;

	};

	__host__ void buildStreetHOST(Hand *reka) {

		reka->rezultat.reszta2=0;
		reka->rezultat.reszta3=0;
		reka->rezultat.reszta4=0;
		reka->rezultat.reszta5=0;

		//char *temp_count = (char*)&fast_var[ threadIdx.x*SHARED_MEM_THREAD_SIZE ];;	
		//__shared__ int temp_count[15 * 16];
		int temp_count[15];
		//offset = 0;
		//offset = threadIdx.x*15;

		for (int i = 0; i < 15; i++)
			temp_count[i] = 0;

		for (int i = 0; i < reka->ile_kart; i++) {
			temp_count[reka->karty[i].wysokosc]++;
		}

		temp_count[1]+=temp_count[14];

		int dno_streeta = 1;
		int szczyt_streeta = 1;

		for (int i = 1; i < 15; i++) {

			if (temp_count[i] > 0)
				szczyt_streeta = i;
			else
				dno_streeta = i + 1;

			if (szczyt_streeta - dno_streeta >= 4) {
				reka->rezultat.reszta1 = i;
			}
		}

		reka->rezultat.poziom = 5;

	};
	
	
	__device__ void buildStreetDEVICE(Hand *reka) {

		reka->rezultat.reszta2=0;
		reka->rezultat.reszta3=0;
		reka->rezultat.reszta4=0;
		reka->rezultat.reszta5=0;

		char *temp_count = (char*)&fast_var[ threadIdx.x*SHARED_MEM_THREAD_SIZE ];;	
		//__shared__ int temp_count[15 * 16];

		//offset = 0;
		//offset = threadIdx.x*15;

		for (int i = 0; i < 15; i++)
			temp_count[i] = 0;

		for (int i = 0; i < reka->ile_kart; i++) {
			temp_count[reka->karty[i].wysokosc]++;
		}

		temp_count[1]+=temp_count[14];

		int dno_streeta = 1;
		int szczyt_streeta = 1;

		for (int i = 1; i < 15; i++) {

			if (temp_count[i] > 0)
				szczyt_streeta = i;
			else
				dno_streeta = i + 1;

			if (szczyt_streeta - dno_streeta >= 4) {
				reka->rezultat.reszta1 = i;
			}
		}

		reka->rezultat.poziom = 5;

	};	
	

	__device__ void buildTrojkaDEVICE(Hand *reka) {


		reka->rezultat.reszta4=0;
		reka->rezultat.reszta5=0;

		char *temp_count = (char*)&fast_var[ threadIdx.x*SHARED_MEM_THREAD_SIZE ];;	
		//__shared__ int temp_count[15 * 16];
	
		//offset = 0;
		//offset = threadIdx.x*15;

		//int count[15];
		int i;
		for (i = 0; i < 15; i++)
			temp_count[i] = 0;

		for (i = 0; i < reka->ile_kart; i++) {
			temp_count[reka->karty[i].wysokosc]++;
		}

		int obliczone;
		obliczone=0;

		for (i = 14; i >= 2; i--) {

			if (temp_count[i] == 3)
				reka->rezultat.reszta1 = i;

			if (temp_count[i] == 1 && obliczone != 2) {
				if (obliczone == 0)
					reka->rezultat.reszta2 = i;
				if (obliczone == 1)
					reka->rezultat.reszta3 = i;

				obliczone++;
			}
		}

		reka->rezultat.poziom = 4;

	};
	__host__ void buildTrojkaHOST(Hand *reka) {


		reka->rezultat.reszta4=0;
		reka->rezultat.reszta5=0;

		//char *temp_count = (char*)&fast_var[ threadIdx.x*SHARED_MEM_THREAD_SIZE ];;	
		int temp_count[15];
	
		//offset = 0;
		//offset = threadIdx.x*15;

		//int count[15];
		int i;
		for (i = 0; i < 15; i++)
			temp_count[i] = 0;

		for (i = 0; i < reka->ile_kart; i++) {
			temp_count[reka->karty[i].wysokosc]++;
		}

		int obliczone;
		obliczone=0;

		for (i = 14; i >= 2; i--) {

			if (temp_count[i] == 3)
				reka->rezultat.reszta1 = i;

			if (temp_count[i] == 1 && obliczone != 2) {
				if (obliczone == 0)
					reka->rezultat.reszta2 = i;
				if (obliczone == 1)
					reka->rezultat.reszta3 = i;

				obliczone++;
			}
		}

		reka->rezultat.poziom = 4;

	};

	__host__ void buildDwieParyHOST(Hand *reka) {

		reka->rezultat.reszta3=0;
		reka->rezultat.reszta4=0;
		reka->rezultat.reszta5=0;

		int temp_count[15];
	
		//offset = 0;
		//offset = threadIdx.x*15;

		//int count[15];
		int i;
		for (i = 0; i < 15; i++)
			temp_count[i] = 0;

		for (i = 0; i < reka->ile_kart; i++)
			temp_count[reka->karty[i].wysokosc]++;

		int obliczone;
		obliczone=0;

		for (i = 14; i >= 2; i--) {

			if (temp_count[i] == 1 && i > reka->rezultat.reszta3)
				reka->rezultat.reszta3 = i;
			if (temp_count[i] == 2 && obliczone == 1) {
				reka->rezultat.reszta2= i;
				obliczone++;
			}
			if (temp_count[i] == 2 && obliczone == 0) {
				reka->rezultat.reszta1 = i;
				obliczone++;
			}

		}

		reka->rezultat.poziom = 3;
	};
	__device__ void buildDwieParyDEVICE(Hand *reka) {

		reka->rezultat.reszta3=0;
		reka->rezultat.reszta4=0;
		reka->rezultat.reszta5=0;

		char *temp_count = (char*)&fast_var[ threadIdx.x*SHARED_MEM_THREAD_SIZE ];;	
		//__shared__ int temp_count[15 * 16];
	
		//offset = 0;
		//offset = threadIdx.x*15;

		//int count[15];
		int i;
		for (i = 0; i < 15; i++)
			temp_count[i] = 0;

		for (i = 0; i < reka->ile_kart; i++)
			temp_count[reka->karty[i].wysokosc]++;

		int obliczone;
		obliczone=0;

		for (i = 14; i >= 2; i--) {

			if (temp_count[i] == 1 && i > reka->rezultat.reszta3)
				reka->rezultat.reszta3 = i;
			if (temp_count[i] == 2 && obliczone == 1) {
				reka->rezultat.reszta2= i;
				obliczone++;
			}
			if (temp_count[i] == 2 && obliczone == 0) {
				reka->rezultat.reszta1 = i;
				obliczone++;
			}

		}

		reka->rezultat.poziom = 3;
	};



	__host__ void buildParaHOST(Hand *reka) {

		int temp_count[15];
	
		//offset = 0;
		//offset = threadIdx.x*15;

		reka->rezultat.reszta5=0;
		//int temp_count[offset +  15];
		int i;
		for (i = 0; i < 15; i++)
			temp_count[i] = 0;

		for (i = 0; i < reka->ile_kart; i++) {
			temp_count[reka->karty[i].wysokosc]++;
		}
		int obliczone;
		obliczone = 0;

		for (i = 14; i >= 2; i--) {

			if (temp_count[i] == 2)
				reka->rezultat.reszta1 = i;
			if (temp_count[i] == 1 && obliczone != 3) {
				if (obliczone==0)
					reka->rezultat.reszta2 = i;
				if (obliczone==1)
					reka->rezultat.reszta3= i;
				if (obliczone==2)
					reka->rezultat.reszta4 = i;

				obliczone++;
			}

		}

		reka->rezultat.poziom = 2;
	};
	__device__ void buildParaDEVICE(Hand *reka) {

		char *temp_count = (char*)&fast_var[ threadIdx.x*SHARED_MEM_THREAD_SIZE ];;	
		//__shared__ int temp_count[15 * 16];
	
		//offset = 0;
		//offset = threadIdx.x*15;

		reka->rezultat.reszta5=0;
		//int temp_count[offset +  15];
		int i;
		for (i = 0; i < 15; i++)
			temp_count[i] = 0;

		for (i = 0; i < reka->ile_kart; i++) {
			temp_count[reka->karty[i].wysokosc]++;
		}
		int obliczone;
		obliczone = 0;

		for (i = 14; i >= 2; i--) {

			if (temp_count[i] == 2)
				reka->rezultat.reszta1 = i;
			if (temp_count[i] == 1 && obliczone != 3) {
				if (obliczone==0)
					reka->rezultat.reszta2 = i;
				if (obliczone==1)
					reka->rezultat.reszta3= i;
				if (obliczone==2)
					reka->rezultat.reszta4 = i;

				obliczone++;
			}

		}

		reka->rezultat.poziom = 2;
	};



	__host__ void buildSmiecHOST(Hand *reka) {

		//char *temp_count = (char*)&fast_var[ threadIdx.x*SHARED_MEM_THREAD_SIZE ];;	
		int temp_count[15 * 16];
	
		//offset = 0;
		//offset = threadIdx.x*15;

		reka->rezultat.poziom = 1;

		//int temp_count[offset +  15];
		int i;
		for (i = 0; i < 15; i++)
			temp_count[ i] = 0;

		for (i = 0; i < reka->ile_kart; i++) {
			temp_count[ reka->karty[i].wysokosc]++;
		}

		int obliczone = 0;

		for (i = 14; i >= 2; i--) {

			if (temp_count[ i] > 0) {
				if (obliczone==0)
					reka->rezultat.reszta1= i;
				if (obliczone==1)
					reka->rezultat.reszta2= i;
				if (obliczone==2)
					reka->rezultat.reszta3 = i;
				if (obliczone==3)
					reka->rezultat.reszta4 = i;
				if (obliczone==4)
					reka->rezultat.reszta5 = i;

				obliczone++;
			}
			if (obliczone == 5)
				return;

		}
	};
	__device__ void buildSmiecDEVICE(Hand *reka) {

		char *temp_count = (char*)&fast_var[ threadIdx.x*SHARED_MEM_THREAD_SIZE ];;	
		//__shared__ int temp_count[15 * 16];
	
		//offset = 0;
		//offset = threadIdx.x*15;

		reka->rezultat.poziom = 1;

		//int temp_count[offset +  15];
		int i;
		for (i = 0; i < 15; i++)
			temp_count[ i] = 0;

		for (i = 0; i < reka->ile_kart; i++) {
			temp_count[ reka->karty[i].wysokosc]++;
		}

		int obliczone = 0;

		for (i = 14; i >= 2; i--) {

			if (temp_count[ i] > 0) {
				if (obliczone==0)
					reka->rezultat.reszta1= i;
				if (obliczone==1)
					reka->rezultat.reszta2= i;
				if (obliczone==2)
					reka->rezultat.reszta3 = i;
				if (obliczone==3)
					reka->rezultat.reszta4 = i;
				if (obliczone==4)
					reka->rezultat.reszta5 = i;

				obliczone++;
			}
			if (obliczone == 5)
				return;

		}
	};


	__host__ int najlepszaKartaHOST(Hand *reka_global) {

		Hand *reka = new Hand();

		for (int i=0; i < 7; i++) {
		  reka->karty[i].wysokosc = reka_global->karty[i].wysokosc;
		  reka->karty[i].kolor = reka_global->karty[i].kolor;
		}
		reka->ile_kart = reka_global->ile_kart;


		int jest_poker;
		jest_poker=0;

		int karty_tej_same_wysokosci[2];
		iloscKartTejSamejWysokosciHOST(reka, &karty_tej_same_wysokosci[0]);

		if (jestKolorHOST(reka) && jestStreetHOST(reka)) {
			jest_poker = jestPokerHOST(reka);
		}

		if (jest_poker) {
			buildPoker(reka_global);
			return 9;
		}
		if (karty_tej_same_wysokosci[0] == 4) {
			buildKareta(reka_global);
			return 8;
		}
		if (karty_tej_same_wysokosci[0] == 3 && karty_tej_same_wysokosci[1]
				>= 2) {
			buildFull(reka_global);
			return 7;
		}
		if (jestKolorHOST(reka)) {
			buildKolor(reka_global);
			return 6;
		}
		if (jestStreetHOST(reka)) {
			buildStreetHOST(reka_global);
			return 5;
		}
		if (karty_tej_same_wysokosci[0] == 3) {
			buildTrojkaHOST(reka_global);
			return 4;
		}
		if (karty_tej_same_wysokosci[0] == 2 && karty_tej_same_wysokosci[1]== 2) {
			buildDwieParyHOST(reka_global);
			return 3;
		}
		if (karty_tej_same_wysokosci[0] == 2) {
			buildParaHOST(reka_global);
			return 2;
		}

		buildSmiecHOST(reka_global);
		return 1;

	};
	__device__ int najlepszaKartaDEVICE(Hand *reka_global) {

		Hand *reka;
		reka = (Hand*)(&fast_var[ threadIdx.x*SHARED_MEM_THREAD_SIZE + 4 ]);
		for (int i=0; i < 7; i++) {
		  reka->karty[i].wysokosc = reka_global->karty[i].wysokosc;
		  reka->karty[i].kolor = reka_global->karty[i].kolor;
		}
		reka->ile_kart = reka_global->ile_kart;


		int jest_poker;
		jest_poker=0;

		int karty_tej_same_wysokosci[2];
		iloscKartTejSamejWysokosciDEVICE(reka, &karty_tej_same_wysokosci[0]);

		if (jestKolorDEVICE(reka) && jestStreetDEVICE(reka)) {
			jest_poker = jestPokerDEVICE(reka);
		}

		if (jest_poker) {
			buildPoker(reka_global);
			return 9;
		}
		if (karty_tej_same_wysokosci[0] == 4) {
			buildKareta(reka_global);
			return 8;
		}
		if (karty_tej_same_wysokosci[0] == 3 && karty_tej_same_wysokosci[1]
				>= 2) {
			buildFull(reka_global);
			return 7;
		}
		if (jestKolorDEVICE(reka)) {
			buildKolor(reka_global);
			return 6;
		}
		if (jestStreetDEVICE(reka)) {
			buildStreetDEVICE(reka_global);
			return 5;
		}
		if (karty_tej_same_wysokosci[0] == 3) {
			buildTrojkaDEVICE(reka_global);
			return 4;
		}
		if (karty_tej_same_wysokosci[0] == 2 && karty_tej_same_wysokosci[1]== 2) {
			buildDwieParyDEVICE(reka_global);
			return 3;
		}
		if (karty_tej_same_wysokosci[0] == 2) {
			buildParaDEVICE(reka_global);
			return 2;
		}

		buildSmiecDEVICE(reka_global);
		return 1;

	};
	





	__host__ void sprawdzRezultatyHOST(Rozdanie *rozdanie) {

		for (int i=0; i < 6; i++) {
			najlepszaKartaHOST(&(rozdanie->handy[i]));
		}

	};
	__device__ void sprawdzRezultatyDEVICE(Rozdanie *rozdanie) {

		for (int i=0; i < 6; i++) {
			najlepszaKartaDEVICE(&(rozdanie->handy[i]));
		}

	};




	__device__ __host__  int porownajRezultaty(Rezultat *rezultat1, Rezultat *rezultat2) {


	//	printf(" rezultat 1 poziom %d ", rezultat1->poziom);
	//	printf(", %d",rezultat1->reszta1);
	//	printf(", %d",rezultat1->reszta2);
	//	printf(", %d",rezultat1->reszta3);
	//	printf(", %d",rezultat1->reszta4);
	//	printf(", %d",rezultat1->reszta5);
	//	printf("\n");
	//
	//	printf(" rezultat 2 poziom %d ", rezultat2->poziom);
	//	printf(", %d",rezultat2->reszta1);
	//	printf(", %d",rezultat2->reszta2);
	//	printf(", %d",rezultat2->reszta3);
	//	printf(", %d",rezultat2->reszta4);
	//	printf(", %d",rezultat2->reszta5);
	//	printf("\n");


		if (rezultat1->poziom < rezultat2->poziom )
			return -1;
		if (rezultat1->poziom > rezultat2->poziom)
			return 1;

		// dla takim samych
		if (rezultat1->reszta1 < rezultat2->reszta1)
			return -1;
		if (rezultat1->reszta1 > rezultat2->reszta1)
			return 1;
		if (rezultat1->reszta2 < rezultat2->reszta2)
			return -1;
		if (rezultat1->reszta2 > rezultat2->reszta2)
			return 1;
		if (rezultat1->reszta3 < rezultat2->reszta3)
			return -1;
		if (rezultat1->reszta3 > rezultat2->reszta3)
			return 1;
		if (rezultat1->reszta4 < rezultat2->reszta4)
			return -1;
		if (rezultat1->reszta4 > rezultat2->reszta4)
			return 1;
		if (rezultat1->reszta5 < rezultat2->reszta5)
			return -1;
		if (rezultat1->reszta5 > rezultat2->reszta5)
			return 1;

		return 0;

	};


	__device__ __host__  int porownaj(Rozdanie *rozdanie, int gracz1, int gracz2) {

	//
	//	for (int i=0; i < 7; i++) {
	//		printf("<%d,",rozdanie->handy[gracz1].karty[i].wysokosc);
	//		printf("%d> ",rozdanie->handy[gracz1].karty[i].kolor);
	//	}
	//	printf("\n");
	//
	//	for (int i=0; i < 7; i++) {
	//		printf("<%d,",rozdanie->handy[gracz2].karty[i].wysokosc);
	//		printf("%d> ",rozdanie->handy[gracz2].karty[i].kolor);
	//	}
	//	printf("\n");


		return porownajRezultaty(
				&rozdanie->handy[gracz1].rezultat,
				&rozdanie->handy[gracz2].rezultat);

	};



	__host__ void najlepszaKartaRozdaniaHOST(Rozdanie *reka, int *wyjscie) {

		for (int i = 0; i < 6; i++) {
			wyjscie[i] = najlepszaKartaHOST(&(reka->handy[i]));
		}

	};
	__device__ void najlepszaKartaRozdaniaDEVICE(Rozdanie *reka, int *wyjscie) {

		for (int i = 0; i < 6; i++) {
			wyjscie[i] = najlepszaKartaDEVICE(&(reka->handy[i]));
		}

	};




	Hand *alokujObiekt(int *karty, int ile_kart) {

		Hand *rezultat = new Hand;
		rezultat->ile_kart = ile_kart;

		rezultat->karty[0].wysokosc = karty[0];
		rezultat->karty[1].wysokosc = karty[1];
		rezultat->karty[2].wysokosc = karty[2];
		rezultat->karty[3].wysokosc = karty[3];
		rezultat->karty[4].wysokosc = karty[4];
		rezultat->karty[5].wysokosc = karty[5];
		rezultat->karty[6].wysokosc = karty[6];

		rezultat->karty[0].kolor = karty[7];
		rezultat->karty[1].kolor = karty[8];
		rezultat->karty[2].kolor = karty[9];
		rezultat->karty[3].kolor = karty[10];
		rezultat->karty[4].kolor = karty[11];
		rezultat->karty[5].kolor = karty[12];
		rezultat->karty[6].kolor = karty[13];

		return rezultat;

	}
	;


}


#endif

