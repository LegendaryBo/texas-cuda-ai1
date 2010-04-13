#include "../struktury/gra.h"
#include "../struktury/gracz.h"

#ifndef GRACZ_AI_H
#define GRACZ_AI_H

extern "C" {


	__host__ void grajRunde1HOST(float bid, IleGracR1 *ile_grac, StawkaR1 *stawka, DobijanieR1 *dobijanie, CzyGracR1 *czy_grac,
			Gra *gra, int ktory_gracz, float *wynik) {

		float wynik_regul[2];

		aplikujCzyGracR1HOST(czy_grac, gra, ktory_gracz, &wynik_regul[0], gra->stawka);

		if ( wynik_regul[0] == 0.0 ) {
			wynik[0] = -1.0;
		} else {

			aplikujStawkaR1HOST(stawka, gra, ktory_gracz, &wynik_regul[0]);

			float prop_stawka = wynik_regul[0];

			if (prop_stawka < gra->stawka) {

				aplikujDobijanieR1HOST(dobijanie, gra, ktory_gracz, &wynik_regul[0], prop_stawka);

				if (wynik_regul[0]==1.0)
					prop_stawka = gra->stawka;
				else
					wynik[0] = -1.0;
			}

			aplikujIleGracR1HOST(ile_grac, gra, ktory_gracz, &wynik_regul[0], prop_stawka);
			wynik[0] = wynik_regul[0];
		}
	};
	__device__ void grajRunde1DEVICE(float bid, IleGracR1 *ile_grac, StawkaR1 *stawka, DobijanieR1 *dobijanie, CzyGracR1 *czy_grac,
			Gra *gra, int ktory_gracz, float *wynik) {

		float wynik_regul[2];

		aplikujCzyGracR1DEVICE(czy_grac, gra, ktory_gracz, &wynik_regul[0], gra->stawka);

		if ( wynik_regul[0] == 0.0 ) {
			wynik[0] = -1.0;
		} else {

			aplikujStawkaR1DEVICE(stawka, gra, ktory_gracz, &wynik_regul[0]);

			float prop_stawka = wynik_regul[0];

			if (prop_stawka < gra->stawka) {

				aplikujDobijanieR1DEVICE(dobijanie, gra, ktory_gracz, &wynik_regul[0], prop_stawka);

				if (wynik_regul[0]==1.0)
					prop_stawka = gra->stawka;
				else
					wynik[0] = -1.0;
			}

			aplikujIleGracR1DEVICE(ile_grac, gra, ktory_gracz, &wynik_regul[0], prop_stawka);
			wynik[0] = wynik_regul[0];
		}
	};	
	


	__host__ void grajRundeXHOST(float bid, IleGracRX *ile_grac, StawkaRX *stawka, DobijanieRX *dobijanie, CzyGracRX *czy_grac,
			Gra *gra, int ktory_gracz, float *wynik) {

		float wynik_regul[2];

		aplikujCzyGracRXHOST(czy_grac, gra, ktory_gracz, &wynik_regul[0], gra->stawka);

		if ( wynik_regul[0] == 0.0 ) {
			wynik[0] = -1.0;
		} else {

			aplikujStawkaRXHOST(stawka, gra, ktory_gracz, &wynik_regul[0]);

			float prop_stawka = wynik_regul[0];

			if (prop_stawka < gra->stawka) {

				aplikujDobijanieRXHOST(dobijanie, gra, ktory_gracz, &wynik_regul[0], prop_stawka);

				if (wynik_regul[0]==1.0)
					prop_stawka = gra->stawka;
				else
					wynik[0] = -1.0;
			}

			aplikujIleGracRXHOST(ile_grac, gra, ktory_gracz, &wynik_regul[0], prop_stawka);

			wynik[0] = wynik_regul[0];
		}
	};
	__device__ void grajRundeXDEVICE(float bid, IleGracRX *ile_grac, StawkaRX *stawka, DobijanieRX *dobijanie, CzyGracRX *czy_grac,
			Gra *gra, int ktory_gracz, float *wynik) {

		float wynik_regul[2];

		aplikujCzyGracRXDEVICE(czy_grac, gra, ktory_gracz, &wynik_regul[0], gra->stawka);

		if ( wynik_regul[0] == 0.0 ) {
			wynik[0] = -1.0;
		} else {

			aplikujStawkaRXDEVICE(stawka, gra, ktory_gracz, &wynik_regul[0]);

			float prop_stawka = wynik_regul[0];

			if (prop_stawka < gra->stawka) {

				aplikujDobijanieRXDEVICE(dobijanie, gra, ktory_gracz, &wynik_regul[0], prop_stawka);

				if (wynik_regul[0]==1.0)
					prop_stawka = gra->stawka;
				else
					wynik[0] = -1.0;
			}

			aplikujIleGracRXDEVICE(ile_grac, gra, ktory_gracz, &wynik_regul[0], prop_stawka);

			wynik[0] = wynik_regul[0];
		}
	};



	Gracz *nowyGracz() {

		return NULL;
	};
};


#endif
