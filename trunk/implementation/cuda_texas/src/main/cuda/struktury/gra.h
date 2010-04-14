#include "gracz.h"
#include "texas_struktury.h"
#include "../struktury/rozdanie.h"

#ifndef GRA_H
#define GRA_H


typedef struct {
	Gracz gracze[6];
	float bids[6];
	int pass[6];

	int mode;
	float minimal_bid;
	float stawka;
	int kto_na_musie;
	int runda;
	float pula;
	int graczyWGrze;
	Rozdanie rozdanie;
	int kto_podbil;

} Gra;


#endif
