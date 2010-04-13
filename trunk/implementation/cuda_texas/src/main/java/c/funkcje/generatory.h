#include "../struktury/texas_struktury.h"
#include "../struktury/rozdanie.h"

#ifndef GENERATORYY_H
#define GENERATORYY_H

extern "C" {

	__device__ __host__ void generuj(int nr_rozdania, Rozdanie *rozdanie, int *seed){

		Karta karty[13*4];

		seed[0] = nr_rozdania;

		for (int i=0; i<13; i++) {
			karty[i].wysokosc=i+2;
			karty[i].kolor=1;

			karty[i +13].wysokosc=i+2;
			karty[i +13].kolor=2;

			karty[i +26].wysokosc=i+2;
			karty[i +26].kolor=3;

			karty[i +39].wysokosc=i+2;
			karty[i +39].kolor=4;
		}

		int liczba_kart=52;
		int wylosowany_index=0;
		int wylosowana_liczba=0;

		// losujemy karty publiczne
		for (int i=0; i < 5; i++) {
			if (seed[0] <0)
				wylosowana_liczba = - seed[0];
			else
				wylosowana_liczba = seed[0];
			wylosowany_index=wylosowana_liczba%liczba_kart;

			rozdanie->karty_publiczne[ i ].wysokosc = karty[wylosowany_index].wysokosc;
			rozdanie->karty_publiczne[ i ].kolor = karty[wylosowany_index].kolor;

			liczba_kart--;
			karty[ wylosowany_index ].wysokosc = karty[liczba_kart].wysokosc;
			karty[ wylosowany_index ].kolor = karty[liczba_kart].kolor;

			seed[0] = (seed[0] * 12991 + 127)%12345789;
		}

		// losujemy karty prywatne
		for (int j=0; j < 6; j++) {
			for (int i=0; i < 2; i++) {
				if (seed[0] <0)
					wylosowana_liczba = - seed[0];
				else
					wylosowana_liczba = seed[0];
				wylosowany_index=wylosowana_liczba%liczba_kart;

				rozdanie->karty_prywatne[ j ][ i ].wysokosc = karty[wylosowany_index].wysokosc;
				rozdanie->karty_prywatne[ j ][ i ].kolor = karty[wylosowany_index].kolor;
				liczba_kart--;
				karty[ wylosowany_index ].wysokosc = karty[liczba_kart].wysokosc;
				karty[ wylosowany_index ].kolor = karty[liczba_kart].kolor;

				seed[0] = (seed[0] * 12991 + 127)%12345789;
			}
		}

		// ustawiamy handsy
		for (int i=0; i < 6; i++) {
			for (int j=0; j<5;j++) {
				rozdanie->handy[i].karty[2+j].wysokosc = rozdanie->karty_publiczne[j].wysokosc;
				rozdanie->handy[i].karty[2+j].kolor = rozdanie->karty_publiczne[j].kolor;
			}

			for (int j=0; j<2;j++) {
				rozdanie->handy[i].karty[j].wysokosc = rozdanie->karty_prywatne[i][j].wysokosc;
				rozdanie->handy[i].karty[j].kolor = rozdanie->karty_prywatne[i][j].kolor;
			}
			rozdanie->handy[i].ile_kart=7;
		}


	//	for (int i=0; i < 6; i++) {
	//		for (int j=0; j < 7; j++) {
	//
	//			printf("<%d",rozdanie->handy[i].karty[j].wysokosc);
	//			printf(",%d>",rozdanie->handy[i].karty[j].kolor);
	//
	//		}
	//		printf("\n");
	//	}

	};

	Rozdanie *gerRozdaniePtr() {

		Rozdanie *rozdanie = new Rozdanie;

		return rozdanie;
	}

};
#endif
