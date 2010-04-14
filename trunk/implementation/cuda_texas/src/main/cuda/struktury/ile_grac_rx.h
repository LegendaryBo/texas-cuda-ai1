#include "kod_graya.h"


#ifndef ILE_GRAC_RX_H
#define ILE_GRAC_RX_H

typedef struct {

	KodGraya gray_graczy_w_grze[5];
	KodGraya gray_graczy_w_grze_fi[5];

	KodGraya gray_stawka[4];
	KodGraya gray_stawka_fi[4];
	KodGraya gray_stawka_wielkosc[4];

	KodGraya gray_pula[4];
	KodGraya gray_pula_fi[4];
	KodGraya gray_pula_wielkosc[4];

	KodGraya gray_rezultat[7];
	KodGraya gray_rezultat_fi[7];

	int dlugosc;
} IleGracRX;

#endif
