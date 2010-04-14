
#ifndef ZLECENIE_H
#define ZLECENIE_H

typedef struct {
	int *osobniki;
	int indexOsobnika[6];
	int indexGracza;
	int nrRozdania;
	float wynik;
} Zlecenie;

#endif
