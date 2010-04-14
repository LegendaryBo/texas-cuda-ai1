#include "../struktury/texas_struktury.h"
#include "../struktury/rozdanie.h"
#include "../struktury/gra.h"
#include "../struktury/kod_graya.h"
#include "../struktury/ile_grac_r1.h"
#include "../struktury/ile_grac_rx.h"

#include "osobnik.h"
#include "rezultaty.h"

#ifndef REGULY_ILEGRAC_H
#define REGULY_ILEGRAC_H

const int DLUGOSC_FI_ILE_GRAC_PARA_R1=5;
const int WIELKOSC_FI_ILE_GRAC_PARA_R1=31;

extern "C" {

	__device__ __host__ void setPula(Gra *gra, int ile) {
		gra->pula = ile;
	};

	__device__ __host__ void setRunda(Gra *gra, int ile) {
		gra->runda = ile;

		for (int i=0; i < 6; i++) {
			if (ile==1)
				gra->rozdanie.handy[i].ile_kart=2;
			if (ile==2)
				gra->rozdanie.handy[i].ile_kart=5;
			if (ile==3)
				gra->rozdanie.handy[i].ile_kart=6;
			if (ile==4)
				gra->rozdanie.handy[i].ile_kart=7;
		}

	};


	void setMode(Gra *gra, int ile) {
		gra->mode = ile;
	}


	__host__ void ileGracParaWRekuR1HOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya_waga, KodGraya *kod_graya_jak_grac,  float *wynik) {

		if (	gra->rozdanie.karty_prywatne[ ktoryGracz ][0].wysokosc
				==
			gra->rozdanie.karty_prywatne[ ktoryGracz ][1].wysokosc) {
			if ( getBitHOST(gra->gracze[ktoryGracz].geny, kod_graya_waga->pozycja_startowa-1) == 1) {
				wynik[0] = (2.0 / WIELKOSC_FI_ILE_GRAC_PARA_R1)
						* obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya_jak_grac );
			}
			else
				wynik[0] = -1.0;

		}
		else
			wynik[0] = -1.0;

		wynik[1] =obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya_waga );
	};
	__device__ void ileGracParaWRekuR1DEVICE(Gra *gra, int ktoryGracz, KodGraya *kod_graya_waga, KodGraya *kod_graya_jak_grac,  float *wynik) {

		if (	gra->rozdanie.karty_prywatne[ ktoryGracz ][0].wysokosc
				==
			gra->rozdanie.karty_prywatne[ ktoryGracz ][1].wysokosc) {
			if ( getBitDEVICE(gra->gracze[ktoryGracz].geny, kod_graya_waga->pozycja_startowa-1) == 1) {
				wynik[0] = (2.0 / WIELKOSC_FI_ILE_GRAC_PARA_R1)
						* obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya_jak_grac );
			}
			else
				wynik[0] = -1.0;

		}
		else
			wynik[0] = -1.0;

		wynik[1] =obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya_waga );
	};
	
	


	__host__ void ileGracKolorWRekuR1HOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya_waga, KodGraya *kod_graya_jak_grac,  float *wynik) {

		if (	gra->rozdanie.karty_prywatne[ ktoryGracz ][0].kolor
				==
			gra->rozdanie.karty_prywatne[ ktoryGracz ][1].kolor) {
			if ( getBitHOST(gra->gracze[ktoryGracz].geny, kod_graya_waga->pozycja_startowa-1) == 1) {
				wynik[0] = (2.0 / WIELKOSC_FI_ILE_GRAC_PARA_R1)
						* obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya_jak_grac );
			}
			else
				wynik[0] = -1.0;
		}
		else
			wynik[0] = -1.0;

		wynik[1] =obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya_waga );
	};
	__device__ void ileGracKolorWRekuR1DEVICE(Gra *gra, int ktoryGracz, KodGraya *kod_graya_waga, KodGraya *kod_graya_jak_grac,  float *wynik) {

		if (	gra->rozdanie.karty_prywatne[ ktoryGracz ][0].kolor
				==
			gra->rozdanie.karty_prywatne[ ktoryGracz ][1].kolor) {
			if ( getBitDEVICE(gra->gracze[ktoryGracz].geny, kod_graya_waga->pozycja_startowa-1) == 1) {
				wynik[0] = (2.0 / WIELKOSC_FI_ILE_GRAC_PARA_R1)
						* obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya_jak_grac );
			}
			else
				wynik[0] = -1.0;

		}
		else
			wynik[0] = -1.0;

		wynik[1] =obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya_waga );
	};
	
	
	
	
	
	
	__host__ void ileGracWysokaKartaWRekuR1HOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya_waga, KodGraya *kod_graya_jak_grac,  float *wynik) {

		if (	gra->rozdanie.karty_prywatne[ ktoryGracz ][0].wysokosc >= 10
				&&
			gra->rozdanie.karty_prywatne[ ktoryGracz ][1].wysokosc >= 10) {
			if ( getBitHOST(gra->gracze[ktoryGracz].geny, kod_graya_waga->pozycja_startowa-1) == 1) {
				wynik[0] = (2.0 / WIELKOSC_FI_ILE_GRAC_PARA_R1)
						* obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya_jak_grac );
			}
			else {
				wynik[0] = -1.0;
			}

		}
		else {
			wynik[0] = -1.0;
		}

		wynik[1] =obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya_waga );
	};
	__device__ void ileGracWysokaKartaWRekuR1DEVICE(Gra *gra, int ktoryGracz, KodGraya *kod_graya_waga, KodGraya *kod_graya_jak_grac,  float *wynik) {

		if (	gra->rozdanie.karty_prywatne[ ktoryGracz ][0].wysokosc >= 10
				&&
			gra->rozdanie.karty_prywatne[ ktoryGracz ][1].wysokosc >= 10) {
			if ( getBitDEVICE(gra->gracze[ktoryGracz].geny, kod_graya_waga->pozycja_startowa-1) == 1) {
				wynik[0] = (2.0 / WIELKOSC_FI_ILE_GRAC_PARA_R1)
						* obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya_jak_grac );
			}
			else {
				wynik[0] = -1.0;
			}

		}
		else {
			wynik[0] = -1.0;
		}

		wynik[1] =obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya_waga );
	};



	__host__ void ileGracBardzoWysokaKartaWRekuR1HOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya_waga, KodGraya *kod_graya_jak_grac,  float *wynik) {

		if (	gra->rozdanie.karty_prywatne[ ktoryGracz ][0].wysokosc >= 13
				&&
			gra->rozdanie.karty_prywatne[ ktoryGracz ][1].wysokosc >= 13) {
			if ( getBitHOST(gra->gracze[ktoryGracz].geny, kod_graya_waga->pozycja_startowa-1) == 1) {
				wynik[0] = (2.0 / WIELKOSC_FI_ILE_GRAC_PARA_R1)
						* obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya_jak_grac );
			}
			else
				wynik[0] = -1.0;
		}
		else
			wynik[0] = -1.0;

		wynik[1] =obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya_waga );
	};
	__device__ void ileGracBardzoWysokaKartaWRekuR1DEVICE(Gra *gra, int ktoryGracz, KodGraya *kod_graya_waga, KodGraya *kod_graya_jak_grac,  float *wynik) {

		if (	gra->rozdanie.karty_prywatne[ ktoryGracz ][0].wysokosc >= 13
				&&
			gra->rozdanie.karty_prywatne[ ktoryGracz ][1].wysokosc >= 13) {
			if ( getBitDEVICE(gra->gracze[ktoryGracz].geny, kod_graya_waga->pozycja_startowa-1) == 1) {
				wynik[0] = (2.0 / WIELKOSC_FI_ILE_GRAC_PARA_R1)
						* obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya_jak_grac );
			}
			else
				wynik[0] = -1.0;

		}
		else
			wynik[0] = -1.0;

		wynik[1] =obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya_waga );
	};



	__host__ void IleGracXGraczyWGrzeRXHOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya_waga, KodGraya *kod_graya_jak_grac,  float *wynik,
			int ile_przeciwnikow) {

		if (	gra->graczyWGrze == ile_przeciwnikow +1 ) {
			if ( getBitHOST(gra->gracze[ktoryGracz].geny, kod_graya_waga->pozycja_startowa-1) == 1) {
				wynik[0] = (2.0 / WIELKOSC_FI_ILE_GRAC_PARA_R1)
						* obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya_jak_grac );
			}
			else
				wynik[0] = -1.0;
		}
		else
			wynik[0] = -1.0;

		wynik[1] =obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya_waga );
	};
	__device__ void IleGracXGraczyWGrzeRXDEVICE(Gra *gra, int ktoryGracz, KodGraya *kod_graya_waga, KodGraya *kod_graya_jak_grac,  float *wynik,
			int ile_przeciwnikow) {

		if (	gra->graczyWGrze == ile_przeciwnikow +1 ) {
			if ( getBitDEVICE(gra->gracze[ktoryGracz].geny, kod_graya_waga->pozycja_startowa-1) == 1) {
				wynik[0] = (2.0 / WIELKOSC_FI_ILE_GRAC_PARA_R1)
						* obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya_jak_grac );
			}
			else
				wynik[0] = -1.0;
		}
		else
			wynik[0] = -1.0;

		wynik[1] =obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya_waga );
	};	
	
	

	__device__ __host__ void setIleGraczyWGrze(Gra *gra, int ile) {
		gra->graczyWGrze = ile;
	};


	__host__ void IleGracPulaRXHOST(Gra *gra, int ktoryGracz,KodGraya *kod_graya_waga, KodGraya *kod_graya_jak_grac, KodGraya *kod_graya_pula, float *wynik) {

		if (	gra->pula >= 100 * gra->minimal_bid * obliczKodGrayaHOST(gra->gracze[ktoryGracz].geny, kod_graya_pula) ) {
			if ( getBitHOST(gra->gracze[ktoryGracz].geny, kod_graya_waga->pozycja_startowa-1) == 1) {
				wynik[0] = (2.0 / WIELKOSC_FI_ILE_GRAC_PARA_R1)
						* obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya_jak_grac );
			}
			else
				wynik[0] = -1.0;

		}
		else
			wynik[0] = -1.0;

		wynik[1] =obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya_waga );
	};
	__device__ void IleGracPulaRXDEVICE(Gra *gra, int ktoryGracz,KodGraya *kod_graya_waga, KodGraya *kod_graya_jak_grac, KodGraya *kod_graya_pula, float *wynik) {

		if (	gra->pula >= 100 * gra->minimal_bid * obliczKodGrayaDEVICE(gra->gracze[ktoryGracz].geny, kod_graya_pula) ) {
			if ( getBitDEVICE(gra->gracze[ktoryGracz].geny, kod_graya_waga->pozycja_startowa-1) == 1) {
				wynik[0] = (2.0 / WIELKOSC_FI_ILE_GRAC_PARA_R1)
						* obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya_jak_grac );
			}
			else
				wynik[0] = -1.0;

		}
		else
			wynik[0] = -1.0;

		wynik[1] =obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya_waga );
	};



	__host__ void IleGracStawkaRXHOST(Gra *gra, int ktoryGracz,KodGraya *kod_graya_waga, KodGraya *kod_graya_jak_grac, KodGraya *kod_graya_pula, float *wynik) {

		if (	gra->stawka >= 10 * gra->minimal_bid * obliczKodGrayaHOST(gra->gracze[ktoryGracz].geny, kod_graya_pula) ) {
			if ( getBitHOST(gra->gracze[ktoryGracz].geny, kod_graya_waga->pozycja_startowa-1) == 1) {
				wynik[0] = (2.0 / WIELKOSC_FI_ILE_GRAC_PARA_R1)
						* obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya_jak_grac );
			}
			else
				wynik[0] = -1.0;

		}
		else
			wynik[0] = -1.0;

		wynik[1] =obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya_waga );

	};
	__device__ void IleGracStawkaRXDEVICE(Gra *gra, int ktoryGracz,KodGraya *kod_graya_waga, KodGraya *kod_graya_jak_grac, KodGraya *kod_graya_pula, float *wynik) {

		if (	gra->stawka >= 10 * gra->minimal_bid * obliczKodGrayaDEVICE(gra->gracze[ktoryGracz].geny, kod_graya_pula) ) {
			if ( getBitDEVICE(gra->gracze[ktoryGracz].geny, kod_graya_waga->pozycja_startowa-1) == 1) {
				wynik[0] = (2.0 / WIELKOSC_FI_ILE_GRAC_PARA_R1)
						* obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya_jak_grac );
			}
			else
				wynik[0] = -1.0;

		}
		else
			wynik[0] = -1.0;

		wynik[1] =obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya_waga );

	};

	__device__ __host__ void setStawka(Gra *gra, int ile) {
		gra->stawka = ile;
	};

	__host__ void IleGracRezultatRXHOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya_waga, KodGraya *kod_graya_jak_grac,  float *wynik, int wymagany_rezultat) {

		int rezultat_gracza = najlepszaKartaHOST( &gra->rozdanie.handy[ktoryGracz] );
		if (	rezultat_gracza >= wymagany_rezultat ) {
			if ( getBitHOST(gra->gracze[ktoryGracz].geny, kod_graya_waga->pozycja_startowa-1) == 1) {
				wynik[0] = (2.0 / WIELKOSC_FI_ILE_GRAC_PARA_R1)
						* obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya_jak_grac );
			}
			else
				wynik[0] = -1.0;

		}
		else
			wynik[0] = -1.0;

		wynik[1] =obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya_waga );
	};
	__device__ void IleGracRezultatRXDEVICE(Gra *gra, int ktoryGracz, KodGraya *kod_graya_waga, KodGraya *kod_graya_jak_grac,  float *wynik, int wymagany_rezultat) {

		int rezultat_gracza = najlepszaKartaDEVICE( &gra->rozdanie.handy[ktoryGracz] );
		if (	rezultat_gracza >= wymagany_rezultat ) {
			if ( getBitDEVICE(gra->gracze[ktoryGracz].geny, kod_graya_waga->pozycja_startowa-1) == 1) {
				wynik[0] = (2.0 / WIELKOSC_FI_ILE_GRAC_PARA_R1)
						* obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya_jak_grac );
			}
			else
				wynik[0] = -1.0;

		}
		else
			wynik[0] = -1.0;

		wynik[1] =obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya_waga );
	};




	__host__ void aplikujIleGracR1HOST(IleGracR1 *reguly, Gra *gra, int ktoryGracz, float *output, float stawka) {

		float suma = 0.0;
		float suma_glosow = 0.0;
		float wyniki_reguly[2];

		ileGracParaWRekuR1HOST(gra, ktoryGracz, &reguly->gray_para_w_reku_waga, &reguly->gray_para_w_reku_fi, &wyniki_reguly[0] );

		suma_glosow += wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
		suma += wyniki_reguly[0] * wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );


		ileGracKolorWRekuR1HOST(gra, ktoryGracz, &reguly->gray_kolor_w_reku_waga, &reguly->gray_kolor_w_reku_fi, &wyniki_reguly[0] );
	
		suma_glosow += wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
		suma += wyniki_reguly[0] * wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );

		ileGracWysokaKartaWRekuR1HOST(gra, ktoryGracz, &reguly->gray_wysoka_karta_w_reku_waga, &reguly->gray_wysoka_karta_w_reku_fi, &wyniki_reguly[0] );

		suma_glosow += wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
		suma += wyniki_reguly[0] * wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );

		ileGracBardzoWysokaKartaWRekuR1HOST(gra, ktoryGracz, &reguly->gray_bardzo_wysoka_karta_w_reku_waga, &reguly->gray_bardzo_wysoka_karta_w_reku_fi, &wyniki_reguly[0] );

		suma_glosow += wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
		suma += wyniki_reguly[0] * wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );


		for (int i=0; i < 5; i++) {
			IleGracXGraczyWGrzeRXHOST(
					gra,
					ktoryGracz,
					&reguly->gray_ile_graczy_waga[i],
					&reguly->gray_ile_graczy_fi[i],
					&wyniki_reguly[0],
					i+1 );


			suma_glosow += wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
			suma += wyniki_reguly[0] * wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
		}

		for (int i=0; i < 4; i++) {
			IleGracStawkaRXHOST(
					gra,
					ktoryGracz,
					&reguly->gray_stawka_waga[i],
					&reguly->gray_stawka_fi[i],
					&reguly->gray_stawka_pula[i],
					&wyniki_reguly[0] );
	
			suma_glosow += wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
			suma += wyniki_reguly[0] * wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );

		}

		if (suma_glosow != 0.0)
			suma = (suma / suma_glosow);

		if (suma * stawka < gra->stawka) {
			if (stawka < gra->minimal_bid) {
				output[0] = gra->minimal_bid;
			} else {
				output[0] = gra->stawka;
			}
		} else {

			if (suma > 1.0) {
				output[0] = gra->stawka;
			} else {
				output[0] = gra->stawka + suma
						* (stawka - gra->stawka);

			}
		}

	};
	__device__ void aplikujIleGracR1DEVICE(IleGracR1 *reguly, Gra *gra, int ktoryGracz, float *output, float stawka) {

		float suma = 0.0;
		float suma_glosow = 0.0;
		float wyniki_reguly[2];

		ileGracParaWRekuR1DEVICE(gra, ktoryGracz, &reguly->gray_para_w_reku_waga, &reguly->gray_para_w_reku_fi, &wyniki_reguly[0] );

		suma_glosow += wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
		suma += wyniki_reguly[0] * wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );


		ileGracKolorWRekuR1DEVICE(gra, ktoryGracz, &reguly->gray_kolor_w_reku_waga, &reguly->gray_kolor_w_reku_fi, &wyniki_reguly[0] );
	
		suma_glosow += wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
		suma += wyniki_reguly[0] * wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );

		ileGracWysokaKartaWRekuR1DEVICE(gra, ktoryGracz, &reguly->gray_wysoka_karta_w_reku_waga, &reguly->gray_wysoka_karta_w_reku_fi, &wyniki_reguly[0] );

		suma_glosow += wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
		suma += wyniki_reguly[0] * wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );

		ileGracBardzoWysokaKartaWRekuR1DEVICE(gra, ktoryGracz, &reguly->gray_bardzo_wysoka_karta_w_reku_waga, &reguly->gray_bardzo_wysoka_karta_w_reku_fi, &wyniki_reguly[0] );

		suma_glosow += wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
		suma += wyniki_reguly[0] * wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );


		for (int i=0; i < 5; i++) {
			IleGracXGraczyWGrzeRXDEVICE(
					gra,
					ktoryGracz,
					&reguly->gray_ile_graczy_waga[i],
					&reguly->gray_ile_graczy_fi[i],
					&wyniki_reguly[0],
					i+1 );


			suma_glosow += wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
			suma += wyniki_reguly[0] * wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
		}

		for (int i=0; i < 4; i++) {
			IleGracStawkaRXDEVICE(
					gra,
					ktoryGracz,
					&reguly->gray_stawka_waga[i],
					&reguly->gray_stawka_fi[i],
					&reguly->gray_stawka_pula[i],
					&wyniki_reguly[0] );
	
			suma_glosow += wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
			suma += wyniki_reguly[0] * wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );

		}

		if (suma_glosow != 0.0)
			suma = (suma / suma_glosow);

		if (suma * stawka < gra->stawka) {
			if (stawka < gra->minimal_bid) {
				output[0] = gra->minimal_bid;
			} else {
				output[0] = gra->stawka;
			}
		} else {

			if (suma > 1.0) {
				output[0] = gra->stawka;
			} else {
				output[0] = gra->stawka + suma
						* (stawka - gra->stawka);

			}
		}

	};




	__host__ void aplikujIleGracRXHOST(IleGracRX *reguly, Gra *gra, int ktoryGracz, float *output, float stawka) {

		float suma = 0.0;
		float suma_glosow = 0.0;
		float wyniki_reguly[2];

		for (int i=1; i <= 5; i++) {
			IleGracXGraczyWGrzeRXHOST(
					gra,
					ktoryGracz,
					&reguly->gray_graczy_w_grze[i-1],
					&reguly->gray_graczy_w_grze_fi[i-1],
					&wyniki_reguly[0],
					i);

			suma_glosow += wyniki_reguly[1] * (wyniki_reguly[0] != -1.0) ;
			suma += wyniki_reguly[0] * wyniki_reguly[1] * (wyniki_reguly[0] != -1.0);
		}

		for (int i=0; i < 4; i++) {
			IleGracStawkaRXHOST(
					gra,
					ktoryGracz,
					&reguly->gray_stawka[i],
					&reguly->gray_stawka_fi[i],
					&reguly->gray_stawka_wielkosc[i],
					&wyniki_reguly[0]);
			suma_glosow += wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
			suma += wyniki_reguly[0] * wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
		}

		for (int i=0; i < 4; i++) {
			IleGracPulaRXHOST(
					gra,
					ktoryGracz,
					&reguly->gray_pula[i],
					&reguly->gray_pula_fi[i],
					&reguly->gray_pula_wielkosc[i],
					&wyniki_reguly[0]);

			suma_glosow += wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
			suma += wyniki_reguly[0] * wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
		}

		for (int i=2; i < 9; i++) {
			IleGracRezultatRXHOST(
					gra,
					ktoryGracz,
					&reguly->gray_rezultat[i-2],
					&reguly->gray_rezultat_fi[i-2],
					&wyniki_reguly[0],
					i);
			suma_glosow += wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
			suma += wyniki_reguly[0] * wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
		}

		if (suma_glosow != 0.0)
			suma = (suma / suma_glosow);
			
		if (suma * stawka < gra->stawka) {
			if (stawka < gra->minimal_bid) {
				output[0] = gra->minimal_bid;
			} else {
				output[0] = gra->stawka;
			}
		} else {

			if (suma > 1.0) {
				output[0] = gra->stawka;
			} else {
				output[0] = gra->stawka + suma
						* (stawka - gra->stawka);

			}
		}


	};
	__device__ void aplikujIleGracRXDEVICE(IleGracRX *reguly, Gra *gra, int ktoryGracz, float *output, float stawka) {

		float suma = 0.0;
		float suma_glosow = 0.0;
		float wyniki_reguly[2];

		for (int i=1; i <= 5; i++) {
			IleGracXGraczyWGrzeRXDEVICE(
					gra,
					ktoryGracz,
					&reguly->gray_graczy_w_grze[i-1],
					&reguly->gray_graczy_w_grze_fi[i-1],
					&wyniki_reguly[0],
					i);

			suma_glosow += wyniki_reguly[1] * (wyniki_reguly[0] != -1.0) ;
			suma += wyniki_reguly[0] * wyniki_reguly[1] * (wyniki_reguly[0] != -1.0);
		}

		for (int i=0; i < 4; i++) {
			IleGracStawkaRXDEVICE(
					gra,
					ktoryGracz,
					&reguly->gray_stawka[i],
					&reguly->gray_stawka_fi[i],
					&reguly->gray_stawka_wielkosc[i],
					&wyniki_reguly[0]);
			suma_glosow += wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
			suma += wyniki_reguly[0] * wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
		}

		for (int i=0; i < 4; i++) {
			IleGracPulaRXDEVICE(
					gra,
					ktoryGracz,
					&reguly->gray_pula[i],
					&reguly->gray_pula_fi[i],
					&reguly->gray_pula_wielkosc[i],
					&wyniki_reguly[0]);

			suma_glosow += wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
			suma += wyniki_reguly[0] * wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
		}

		for (int i=2; i < 9; i++) {
			IleGracRezultatRXDEVICE(
					gra,
					ktoryGracz,
					&reguly->gray_rezultat[i-2],
					&reguly->gray_rezultat_fi[i-2],
					&wyniki_reguly[0],
					i);
			suma_glosow += wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
			suma += wyniki_reguly[0] * wyniki_reguly[1] * (wyniki_reguly[0] != -1.0 );
		}

		if (suma_glosow != 0.0)
			suma = (suma / suma_glosow);
			
		if (suma * stawka < gra->stawka) {
			if (stawka < gra->minimal_bid) {
				output[0] = gra->minimal_bid;
			} else {
				output[0] = gra->stawka;
			}
		} else {

			if (suma > 1.0) {
				output[0] = gra->stawka;
			} else {
				output[0] = gra->stawka + suma
						* (stawka - gra->stawka);

			}
		}


	};



	void getIleGracRXPTR(int przesuniecie, IleGracRX *reguly) {


		for (int i=1; i <= 5; i++) {
			reguly->gray_graczy_w_grze[i-1].pozycja_startowa  =  przesuniecie + 1;
			reguly->gray_graczy_w_grze[i-1].dlugosc = 8;
			przesuniecie += 1+8;
			reguly->gray_graczy_w_grze_fi[i-1].pozycja_startowa  =  przesuniecie;
			reguly->gray_graczy_w_grze_fi[i-1].dlugosc = 5;
			przesuniecie += 5;
		}

		for (int i=0; i < 4; i++) {
			reguly->gray_stawka[i].pozycja_startowa  =  przesuniecie + 1;
			reguly->gray_stawka[i].dlugosc = 8;
			przesuniecie += 1+8;
			reguly->gray_stawka_fi[i].pozycja_startowa  =  przesuniecie;
			reguly->gray_stawka_fi[i].dlugosc = 5;
			przesuniecie += 5;
			reguly->gray_stawka_wielkosc[i].pozycja_startowa  =  przesuniecie;
			reguly->gray_stawka_wielkosc[i].dlugosc = 10;
			przesuniecie += 10;
		}

		for (int i=0; i < 4; i++) {
			reguly->gray_pula[i].pozycja_startowa  =  przesuniecie + 1;
			reguly->gray_pula[i].dlugosc = 8;
			przesuniecie += 1+8;
			reguly->gray_pula_fi[i].pozycja_startowa  =  przesuniecie;
			reguly->gray_pula_fi[i].dlugosc = 5;
			przesuniecie += 5;
			reguly->gray_pula_wielkosc[i].pozycja_startowa  =  przesuniecie;
			reguly->gray_pula_wielkosc[i].dlugosc = 10;
			przesuniecie += 10;
		}

		for (int i=2; i < 9; i++) {
			reguly->gray_rezultat[i-2].pozycja_startowa  =  przesuniecie + 1;
			reguly->gray_rezultat[i-2].dlugosc = 8;
			przesuniecie += 1+8;
			reguly->gray_rezultat_fi[i-2].pozycja_startowa  =  przesuniecie;
			reguly->gray_rezultat_fi[i-2].dlugosc = 5;
			przesuniecie += 5;
		}


	};

	void getIleGracR1PTR(int przesuniecie, IleGracR1 *ile_grac) {

		ile_grac->gray_para_w_reku_waga.pozycja_startowa = przesuniecie + 1;
		ile_grac->gray_para_w_reku_waga.dlugosc = 8;
		ile_grac->gray_para_w_reku_fi.pozycja_startowa = przesuniecie + 1 + 8;
		ile_grac->gray_para_w_reku_fi.dlugosc = 5;
		przesuniecie += 1+5+8;

		ile_grac->gray_kolor_w_reku_waga.pozycja_startowa = przesuniecie + 1;
		ile_grac->gray_kolor_w_reku_waga.dlugosc = 8;
		ile_grac->gray_kolor_w_reku_fi.pozycja_startowa = przesuniecie + 1 + 8;
		ile_grac->gray_kolor_w_reku_fi.dlugosc = 5;
		przesuniecie += 1+5+8;

		ile_grac->gray_wysoka_karta_w_reku_waga.pozycja_startowa = przesuniecie + 1;
		ile_grac->gray_wysoka_karta_w_reku_waga.dlugosc = 8;
		ile_grac->gray_wysoka_karta_w_reku_fi.pozycja_startowa = przesuniecie + 1 + 8;
		ile_grac->gray_wysoka_karta_w_reku_fi.dlugosc = 5;
		przesuniecie += 1+5+8;

		ile_grac->gray_bardzo_wysoka_karta_w_reku_waga.pozycja_startowa = przesuniecie + 1;
		ile_grac->gray_bardzo_wysoka_karta_w_reku_waga.dlugosc = 8;
		ile_grac->gray_bardzo_wysoka_karta_w_reku_fi.pozycja_startowa = przesuniecie + 1 + 8;
		ile_grac->gray_bardzo_wysoka_karta_w_reku_fi.dlugosc = 5;
		przesuniecie += 1+5+8;

		for (int i=0; i < 5; i++) {
			ile_grac->gray_ile_graczy_waga[i].pozycja_startowa  =  przesuniecie + 1;
			ile_grac->gray_ile_graczy_waga[i].dlugosc = 8;
			ile_grac->gray_ile_graczy_fi[i].pozycja_startowa = przesuniecie + 1+8;
			ile_grac->gray_ile_graczy_fi[i].dlugosc = 5;

			przesuniecie += 1+5+8;
		}

		for (int i=0; i < 4; i++) {
			ile_grac->gray_stawka_waga[i].pozycja_startowa = przesuniecie + 1;
			ile_grac->gray_stawka_waga[i].dlugosc = 8;
			ile_grac->gray_stawka_fi[i].pozycja_startowa = przesuniecie + 1+8;
			ile_grac->gray_stawka_fi[i].dlugosc = 5;
			ile_grac->gray_stawka_pula[i].pozycja_startowa = przesuniecie + 1+8+5;
			ile_grac->gray_stawka_pula[i].dlugosc = 10;
			przesuniecie += 1+5+8+10;
		}


	};
}
#endif
