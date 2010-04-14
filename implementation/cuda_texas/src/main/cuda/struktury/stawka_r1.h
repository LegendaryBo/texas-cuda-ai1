#include "kod_graya.h"


#ifndef STAWKA_R1_H
#define STAWKA_R1_H

typedef struct {

	KodGraya gray_para;
	KodGraya gray_kolor;
	KodGraya gray_kolor2;
	KodGraya gray_wysokie_karty;
	KodGraya gray_bardzo_wysokie_karty;
	KodGraya gray_stala_stawka_mala;
	KodGraya gray_stala_stawka_srednia;
	KodGraya gray_stala_stawka_duza;

	int dlugosc;
} StawkaR1;

#endif
