#include "kod_graya.h"


#ifndef STAWKA_RX_H
#define STAWKA_RX_H

typedef struct {

	KodGraya gray_jest_rezultat[8];

	KodGraya gray_mala_stawka[5];
	KodGraya gray_mala_stawka_parametr[5];

	int dlugosc;
} StawkaRX;

#endif
