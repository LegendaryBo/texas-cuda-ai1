//#include <iostream>

#ifndef TEXAS_STRUKTURY_H
#define TEXAS_STRUKTURY_H

typedef struct {
	char wysokosc;
	char kolor;
} Karta;


typedef struct {
	char poziom;
	char reszta1;
	char reszta2;
	char reszta3;
	char reszta4;
	char reszta5;
} Rezultat;



Karta *getLosowaKarta();


#endif
