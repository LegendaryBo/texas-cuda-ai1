#include "kod_graya.h"


#ifndef CZY_GRAC_RX_H
#define CZY_GRAC_RX_H

typedef struct {

	KodGraya gray_na_wejscie;

	KodGraya gray_rezultat[7];

	KodGraya gray_ograniczenie_stawki[5];
	KodGraya gray_ograniczenie_stawki_stawka[5];


	int dlugosc;
} CzyGracRX;

#endif
