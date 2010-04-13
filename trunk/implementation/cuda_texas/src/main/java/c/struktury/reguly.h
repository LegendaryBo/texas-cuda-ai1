#include "ile_grac_r1.h"
#include "ile_grac_rx.h"
#include "czy_grac_r1.h"
#include "czy_grac_rx.h"
#include "dobijanie_r1.h"
#include "dobijanie_rx.h"
#include "stawka_r1.h"
#include "stawka_rx.h"


#ifndef REGULY_H
#define REGULY_H

typedef struct {

	IleGracR1 ile_grac_r1;
	IleGracRX ile_grac_rx[3];

	StawkaR1 stawka_r1;
	StawkaRX stawka_rx[3];


	DobijanieR1 dobijanie_r1;
	DobijanieRX dobijanie_rx[3];


	CzyGracR1 czy_grac_r1;
	CzyGracRX czy_grac_rx[3];


} Reguly;

#endif
