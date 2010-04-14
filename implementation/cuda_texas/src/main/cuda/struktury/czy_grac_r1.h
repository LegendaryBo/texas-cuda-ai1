#include "kod_graya.h"


#ifndef CZY_GRAC_R1_H
#define CZY_GRAC_R1_H

typedef struct {

	KodGraya gray_wymaganych_glosow;

	KodGraya gray_para_w_rece;
	KodGraya gray_kolor_w_rece;

	KodGraya gray_ograniczenie_stawki;
	KodGraya gray_ograniczenie_stawki_wielkosc;

	KodGraya gray_wysoka_karta_w_rece;
	KodGraya gray_bardzo_wysoka_karta_w_rece;

	int dlugosc;
} CzyGracR1;

#endif
