#include "../struktury/kod_graya.h"

#ifndef OSOBNIK_H
#define OSOBNIK_H

extern "C" {

	__host__  int getBitHOST(int *osobnik, int ktory_bit) {
		return ( (unsigned int) (osobnik[ktory_bit / 32] & (1 << ktory_bit % 32)) )>> (ktory_bit
				% 32);
	};
	__device__  int getBitDEVICE(int *osobnik, int ktory_bit) {
		return ( (unsigned int) (osobnik[ktory_bit / 32] & (1 << ktory_bit % 32)) )>> (ktory_bit
				% 32);
	};

	__host__ int obliczKodGrayaHOST(int *osobnik, KodGraya *kod_graya) {

		int pLiczbaBinarna = 0;
		int pBit = getBitHOST(osobnik, kod_graya->pozycja_startowa);
		pLiczbaBinarna += pBit << kod_graya->dlugosc - 1;

		for (int i = 1; i < kod_graya->dlugosc; i++) {
			pBit = (getBitHOST(osobnik, kod_graya->pozycja_startowa + i) ^ pBit);
			pLiczbaBinarna += pBit << (kod_graya->dlugosc - i - 1);
		}
		return pLiczbaBinarna;
	};

	__device__ int obliczKodGrayaDEVICE(int *osobnik, KodGraya *kod_graya) {

		int pLiczbaBinarna = 0;
		int pBit = getBitDEVICE(osobnik, kod_graya->pozycja_startowa);
		pLiczbaBinarna += pBit << kod_graya->dlugosc - 1;

		for (int i = 1; i < kod_graya->dlugosc; i++) {
			pBit = (getBitDEVICE(osobnik, kod_graya->pozycja_startowa + i) ^ pBit);
			pLiczbaBinarna += pBit << (kod_graya->dlugosc - i - 1);
		}
		return pLiczbaBinarna;
	};

	KodGraya  *getKodGrayaPTR(int pozycja_startowa, int dlugosc) {

		KodGraya *kod_graya = new  KodGraya;

		kod_graya->pozycja_startowa = pozycja_startowa;
		kod_graya->dlugosc = dlugosc;

		return kod_graya;
	}
	;


	int *getOsobnikPTR(int *geny, int dlugosc_intow) {

		int *osobnik = (int*)malloc(sizeof(geny) * dlugosc_intow);

		for (int i=0; i < dlugosc_intow; i++) {
			osobnik[i] = geny[i];
		}

		return osobnik;
	}
	;

}


#endif
