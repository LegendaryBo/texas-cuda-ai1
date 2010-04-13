#include "../struktury/reguly.h"


#ifndef REGULY_FUNKCJE_H
#define REGULY_FUNKCJE_H


int PRZESUNIECIE_CZYGRAC_R1=0;
int PRZESUNIECIE_STAWKA_R1=41;
int PRZESUNIECIE_DOBIJANIE_R1=106;
int PRZESUNIECIE_ILEGRAC_R1=118;

int PRZESUNIECIE_CZYGRAC_R2=340;
int PRZESUNIECIE_STAWKA_R2=489;
int PRZESUNIECIE_DOBIJANIE_R2=631;
int PRZESUNIECIE_ILEGRAC_R2=643;

int PRZESUNIECIE_CZYGRAC_R3=1003;
int PRZESUNIECIE_STAWKA_R3=1152;
int PRZESUNIECIE_DOBIJANIE_R3=1294;
int PRZESUNIECIE_ILEGRAC_R3=1306;

int PRZESUNIECIE_CZYGRAC_R4=1666;
int PRZESUNIECIE_STAWKA_R4=1815;
int PRZESUNIECIE_DOBIJANIE_R4=1957;
int PRZESUNIECIE_ILEGRAC_R4=1969;


extern "C" {


__host__ Reguly *getReguly() {

	Reguly *reguly = new Reguly();

	getCzyGracR1PTR(PRZESUNIECIE_CZYGRAC_R1, &reguly->czy_grac_r1) ;
	getStawkaR1PTR(PRZESUNIECIE_STAWKA_R1, &reguly->stawka_r1) ;
	getDobijanieR1PTR(PRZESUNIECIE_DOBIJANIE_R1, &reguly->dobijanie_r1) ;
	getIleGracR1PTR(PRZESUNIECIE_ILEGRAC_R1, &reguly->ile_grac_r1) ;
//
	getCzyGracRXPTR(PRZESUNIECIE_CZYGRAC_R2, &reguly->czy_grac_rx[0]) ;
	getStawkaRXPTR(PRZESUNIECIE_STAWKA_R2, &reguly->stawka_rx[0]) ;
	getDobijanieRXPTR(PRZESUNIECIE_DOBIJANIE_R2, &reguly->dobijanie_rx[0]) ;
	getIleGracRXPTR(PRZESUNIECIE_ILEGRAC_R2, &reguly->ile_grac_rx[0]) ;
//
	getCzyGracRXPTR(PRZESUNIECIE_CZYGRAC_R3, &reguly->czy_grac_rx[1]) ;
	getStawkaRXPTR(PRZESUNIECIE_STAWKA_R3, &reguly->stawka_rx[1]) ;
	getDobijanieRXPTR(PRZESUNIECIE_DOBIJANIE_R3, &reguly->dobijanie_rx[1]) ;
	getIleGracRXPTR(PRZESUNIECIE_ILEGRAC_R3, &reguly->ile_grac_rx[1]) ;
//
	getCzyGracRXPTR(PRZESUNIECIE_CZYGRAC_R4, &reguly->czy_grac_rx[2]) ;
	getStawkaRXPTR(PRZESUNIECIE_STAWKA_R4, &reguly->stawka_rx[2]) ;
	getDobijanieRXPTR(PRZESUNIECIE_DOBIJANIE_R4, &reguly->dobijanie_rx[2]) ;
	getIleGracRXPTR(PRZESUNIECIE_ILEGRAC_R4, &reguly->ile_grac_rx[2]) ;

	return reguly;
};


IleGracR1 *getIleGracR1PTRZReguly(Reguly *reguly) {
	return &reguly->ile_grac_r1;
};

StawkaR1 *getStawkaR1PTRZReguly(Reguly *reguly) {
	return &reguly->stawka_r1;
};

DobijanieR1 *getDobijanieR1PTRZReguly(Reguly *reguly) {
	return &reguly->dobijanie_r1;
};

CzyGracR1 *getCzyGracR1PTRZReguly(Reguly *reguly) {
	return &reguly->czy_grac_r1;
};

IleGracRX *getIleGracRXPTRZReguly(Reguly *reguly, int runda) {
	return &reguly->ile_grac_rx[runda-2];
};

StawkaRX *getStawkaRXPTRZReguly(Reguly *reguly, int runda) {
	return &reguly->stawka_rx[runda-2];
};

DobijanieRX *getDobijanieRXPTRZReguly(Reguly *reguly, int runda) {
	return &reguly->dobijanie_rx[runda-2];
};

CzyGracRX *getCzyGracRXPTRZReguly(Reguly *reguly, int runda) {
	return &reguly->czy_grac_rx[runda-2];
};

}

#endif
