#include "hand.h"
#include "texas_struktury.h"

#ifndef ROZDANIE_H
#define ROZDANIE_H

typedef struct {
	Karta karty_prywatne[6][2];
	Karta karty_publiczne[5];
	Hand handy[6];
} Rozdanie;

#endif
