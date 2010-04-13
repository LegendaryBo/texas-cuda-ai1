#include "kod_graya.h"


#ifndef DOBIJANIE_R1_H
#define DOBIJANIE_R1_H

typedef struct {

	int dobijac_zawsze;
	int dobijac_brakuje_x[5];
/*
	int dobijac_brakuje_x_06;
	int dobijac_brakuje_x_08;
	int dobijac_brakuje_x_01;
	int dobijac_brakuje_x_032;
*/

	int dobijac_para_w_rece_01;
	int dobijac_para_w_rece_03;
	int dobijac_para_w_rece_06;

	int dobijac_wysoka_reka_01;
	int dobijac_wysoka_reka_03;
	int dobijac_wysoka_reka_06;

	int dlugosc;
} DobijanieR1;

#endif
