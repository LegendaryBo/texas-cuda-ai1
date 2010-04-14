#include "kod_graya.h"


#ifndef ILE_GRAC_R1_H
#define ILE_GRAC_R1_H

typedef struct {

	KodGraya gray_para_w_reku_waga;
	KodGraya gray_para_w_reku_fi;

	KodGraya gray_kolor_w_reku_waga;
	KodGraya gray_kolor_w_reku_fi;

	KodGraya gray_wysoka_karta_w_reku_waga;
	KodGraya gray_wysoka_karta_w_reku_fi;

	KodGraya gray_bardzo_wysoka_karta_w_reku_waga;
	KodGraya gray_bardzo_wysoka_karta_w_reku_fi;

	KodGraya gray_ile_graczy_waga[5];
	KodGraya gray_ile_graczy_fi[5];

	KodGraya gray_stawka_waga[4];
	KodGraya gray_stawka_fi[4];
	KodGraya gray_stawka_pula[4];

	int dlugosc;
} IleGracR1;

#endif
