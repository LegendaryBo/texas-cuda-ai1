#include "../struktury/texas_struktury.h"
#include "../struktury/rozdanie.h"
#include "../struktury/gra.h"
#include "../struktury/kod_graya.h"
#include "../struktury/czy_grac_r1.h"
#include "../struktury/czy_grac_rx.h"

#include "osobnik.h"
#include "rezultaty.h"

#ifndef REGULY_CZYGRAC_H
#define REGULY_CZYGRAC_H

extern "C" {





	__device__ void grajGdyParaWRekuR1DEVICE(Gra *gra, int ktoryGracz,  KodGraya *kod_graya,  float *wynik, float stawka) {
		wynik[0] = (gra->rozdanie.karty_prywatne[ ktoryGracz ][0].wysokosc == gra->rozdanie.karty_prywatne[ ktoryGracz ][1].wysokosc)
			    * ( getBitDEVICE(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1)
			    * (float)obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya );
	};
	__host__ void grajGdyParaWRekuR1HOST(Gra *gra, int ktoryGracz,  KodGraya *kod_graya,  float *wynik, float stawka) {
		wynik[0] = (gra->rozdanie.karty_prywatne[ ktoryGracz ][0].wysokosc == gra->rozdanie.karty_prywatne[ ktoryGracz ][1].wysokosc)
			    * ( getBitHOST(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1)
			    * (float)obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya );
	};

	__device__ void grajGdyKolorWRekuR1DEVICE(Gra *gra, int ktoryGracz,  KodGraya *kod_graya,  float *wynik, float stawka) {
		wynik[0] = (gra->rozdanie.karty_prywatne[ ktoryGracz ][0].kolor==gra->rozdanie.karty_prywatne[ ktoryGracz ][1].kolor)
			    * (getBitDEVICE(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1)
			    * (float)obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya );
	};
	__host__ void grajGdyKolorWRekuR1HOST(Gra *gra, int ktoryGracz,  KodGraya *kod_graya,  float *wynik, float stawka) {
		wynik[0] = (gra->rozdanie.karty_prywatne[ ktoryGracz ][0].kolor==gra->rozdanie.karty_prywatne[ ktoryGracz ][1].kolor)
			    * (getBitHOST(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1)
			    * (float)obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya );
	};


	__device__ void grajWysokieKartyNaWejscieR1DEVICE(Gra *gra, int ktoryGracz,  KodGraya *kod_graya,  float *wynik, float stawka){
		wynik[0] = (gra->rozdanie.karty_prywatne[ ktoryGracz ][0].wysokosc > 10&&gra->rozdanie.karty_prywatne[ ktoryGracz ][1].wysokosc > 10)
			    * ( getBitDEVICE(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1)
			    * (float)obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya);
	};
	__host__ void grajWysokieKartyNaWejscieR1HOST(Gra *gra, int ktoryGracz,  KodGraya *kod_graya,  float *wynik, float stawka){
		wynik[0] = (gra->rozdanie.karty_prywatne[ ktoryGracz ][0].wysokosc > 10&&gra->rozdanie.karty_prywatne[ ktoryGracz ][1].wysokosc > 10)
			    * ( getBitHOST(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1)
			    * (float)obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya);
	};



	__device__ void grajBardzoWysokieKartyNaWejscieR1DEVICE(Gra *gra, int ktoryGracz,  KodGraya *kod_graya,  float *wynik, float stawka){
	      wynik[0] = (gra->rozdanie.karty_prywatne[ ktoryGracz ][0].wysokosc > 12&&gra->rozdanie.karty_prywatne[ ktoryGracz ][1].wysokosc > 12)
			  * ( getBitDEVICE(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1)
			  * (float)obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya);
	};
	__host__ void grajBardzoWysokieKartyNaWejscieR1HOST(Gra *gra, int ktoryGracz,  KodGraya *kod_graya,  float *wynik, float stawka){
	      wynik[0] = (gra->rozdanie.karty_prywatne[ ktoryGracz ][0].wysokosc > 12&&gra->rozdanie.karty_prywatne[ ktoryGracz ][1].wysokosc > 12)
			  * ( getBitHOST(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1)
			  * (float)obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya);
	};



	__device__ void wymaganychGlosowRXDEVICE(Gra *gra, int ktoryGracz,  KodGraya *kod_graya,  float *wynik) {
		wynik[0] = (float)obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya);
	};
	__host__ void wymaganychGlosowRXHOST(Gra *gra, int ktoryGracz,  KodGraya *kod_graya,  float *wynik) {
		wynik[0] = (float)obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya);
	};






	__device__ void grajRezultatRXDEVICE(Gra *gra, int ktoryGracz,  KodGraya *kod_graya,  float *wynik,  int wymagany_rezultat) {

		int rezultat_gracza = najlepszaKartaDEVICE( &gra->rozdanie.handy[ktoryGracz] );
		if (	rezultat_gracza >= wymagany_rezultat) {
			if ( getBitDEVICE(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1) {
				wynik[0] = (float)obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya);
			}
			else {
				wynik[0] = 0.0;
			}
		}
		else {
			wynik[0] = 0.0;
		}
	};
	__host__ void grajRezultatRXHOST(Gra *gra, int ktoryGracz,  KodGraya *kod_graya,  float *wynik,  int wymagany_rezultat) {

		int rezultat_gracza = najlepszaKartaHOST( &gra->rozdanie.handy[ktoryGracz] );
		if (	rezultat_gracza >= wymagany_rezultat) {
			if ( getBitHOST(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1) {
				wynik[0] = (float)obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya);
			}
			else {
				wynik[0] = 0.0;
			}
		}
		else {
			wynik[0] = 0.0;
		}
	};




	__host__ void grajOgraniczenieStawkiNaWejscieR1HOST(Gra *gra, int ktoryGracz,  KodGraya *kod_graya,  KodGraya *kod_graya2,  float *wynik) {
	      wynik[0] = (getBitHOST(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1 )
			  * (gra->stawka <= (float)obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya2 ) )
			  * (float)obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya );
	};
	__device__ void grajOgraniczenieStawkiNaWejscieR1DEVICE(Gra *gra, int ktoryGracz,  KodGraya *kod_graya,  KodGraya *kod_graya2,  float *wynik) {
	      wynik[0] = (getBitDEVICE(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1 )
			  * (gra->stawka <= (float)obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya2 ) )
			  * (float)obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya );
	};






	__host__ void ograniczenieStawkiRXHOST(Gra *gra, int ktoryGracz,  KodGraya *kod_graya, KodGraya *kod_graya2,  float *wynik) {
		wynik[0] = (getBitHOST(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1)
			    * (gra->stawka <= (float)obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya2))
			    * (float)obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya);
	};
	__device__ void ograniczenieStawkiRXDEVICE(Gra *gra, int ktoryGracz,  KodGraya *kod_graya, KodGraya *kod_graya2,  float *wynik) {
		wynik[0] = (getBitDEVICE(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1)
			    * (gra->stawka <= (float)obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya2))
			    * (float)obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya);
	};	
	
	


	__host__  void aplikujCzyGracR1HOST(CzyGracR1 *reguly, Gra *gra, int ktoryGracz, float *output, float stawka) {

		float wynik_reguly[1];

		wymaganychGlosowRXHOST(gra, ktoryGracz,  &reguly->gray_wymaganych_glosow,  &wynik_reguly[0]);

		float wymaganych_glosow = wynik_reguly[0];
		float glosow = 0.0;

		grajGdyParaWRekuR1HOST(gra, ktoryGracz, &reguly->gray_para_w_rece, &wynik_reguly[0], stawka );
		glosow += wynik_reguly[0];

		grajGdyKolorWRekuR1HOST(gra, ktoryGracz, &reguly->gray_kolor_w_rece, &wynik_reguly[0], stawka );
		glosow += wynik_reguly[0];

		grajOgraniczenieStawkiNaWejscieR1HOST(gra, ktoryGracz, &reguly->gray_ograniczenie_stawki, &reguly->gray_ograniczenie_stawki_wielkosc, &wynik_reguly[0]);
		glosow += wynik_reguly[0];

		grajWysokieKartyNaWejscieR1HOST(gra, ktoryGracz, &reguly->gray_wysoka_karta_w_rece, &wynik_reguly[0], stawka );
		glosow += wynik_reguly[0];

		grajBardzoWysokieKartyNaWejscieR1HOST(gra, ktoryGracz, &reguly->gray_bardzo_wysoka_karta_w_rece, &wynik_reguly[0], stawka );
		glosow += wynik_reguly[0];

		output[0] = 1.0 * (glosow >= wymaganych_glosow);
	};
	__device__  void aplikujCzyGracR1DEVICE(CzyGracR1 *reguly, Gra *gra, int ktoryGracz, float *output, float stawka) {

		float wynik_reguly[1];

		wymaganychGlosowRXDEVICE(gra, ktoryGracz,  &reguly->gray_wymaganych_glosow,  &wynik_reguly[0]);

		float wymaganych_glosow = wynik_reguly[0];
		float glosow = 0.0;

		grajGdyParaWRekuR1DEVICE(gra, ktoryGracz, &reguly->gray_para_w_rece, &wynik_reguly[0], stawka );
		glosow += wynik_reguly[0];

		grajGdyKolorWRekuR1DEVICE(gra, ktoryGracz, &reguly->gray_kolor_w_rece, &wynik_reguly[0], stawka );
		glosow += wynik_reguly[0];

		grajOgraniczenieStawkiNaWejscieR1DEVICE(gra, ktoryGracz, &reguly->gray_ograniczenie_stawki, &reguly->gray_ograniczenie_stawki_wielkosc, &wynik_reguly[0]);
		glosow += wynik_reguly[0];

		grajWysokieKartyNaWejscieR1DEVICE(gra, ktoryGracz, &reguly->gray_wysoka_karta_w_rece, &wynik_reguly[0], stawka );
		glosow += wynik_reguly[0];

		grajBardzoWysokieKartyNaWejscieR1DEVICE(gra, ktoryGracz, &reguly->gray_bardzo_wysoka_karta_w_rece, &wynik_reguly[0], stawka );
		glosow += wynik_reguly[0];

		output[0] = 1.0 * (glosow >= wymaganych_glosow);
	};



	__host__ void aplikujCzyGracRXHOST(CzyGracRX *reguly, Gra *gra, int ktoryGracz, float *output, float stawka) {

		float wynik_reguly[1];

		wymaganychGlosowRXHOST(gra, ktoryGracz,  &reguly->gray_na_wejscie,  &wynik_reguly[0]);

		float wymaganych_glosow = wynik_reguly[0];
		float glosow = 0.0;

		for (int i=2; i < 9; i++) {
			grajRezultatRXHOST(gra, ktoryGracz, &reguly->gray_rezultat[i-2], &wynik_reguly[0], i );
			glosow += wynik_reguly[0];
		}

		for (int i=0; i < 5; i++) {
			ograniczenieStawkiRXHOST(gra, ktoryGracz, &reguly->gray_ograniczenie_stawki[i], &reguly->gray_ograniczenie_stawki_stawka[i],  &wynik_reguly[0]);
			glosow += wynik_reguly[0];
		}

		output[0] = 1.0 * (glosow >= wymaganych_glosow);
	};
	__device__ void aplikujCzyGracRXDEVICE(CzyGracRX *reguly, Gra *gra, int ktoryGracz, float *output, float stawka) {

		float wynik_reguly[1];

		wymaganychGlosowRXDEVICE(gra, ktoryGracz,  &reguly->gray_na_wejscie,  &wynik_reguly[0]);

		float wymaganych_glosow = wynik_reguly[0];
		float glosow = 0.0;

		for (int i=2; i < 9; i++) {
			grajRezultatRXDEVICE(gra, ktoryGracz, &reguly->gray_rezultat[i-2], &wynik_reguly[0], i );
			glosow += wynik_reguly[0];
		}

		for (int i=0; i < 5; i++) {
			ograniczenieStawkiRXDEVICE(gra, ktoryGracz, &reguly->gray_ograniczenie_stawki[i], &reguly->gray_ograniczenie_stawki_stawka[i],  &wynik_reguly[0]);
			glosow += wynik_reguly[0];
		}

		output[0] = 1.0 * (glosow >= wymaganych_glosow);
	};




	void getCzyGracRXPTR(int przesuniecie, CzyGracRX *reguly) {

		reguly->gray_na_wejscie.pozycja_startowa = przesuniecie;
		reguly->gray_na_wejscie.dlugosc = 7;
		przesuniecie += 7;


		for (int i=2; i < 9; i++) {
			reguly->gray_rezultat[i-2].pozycja_startowa = przesuniecie + 1;
			reguly->gray_rezultat[i-2].dlugosc = 5+i;
			przesuniecie += 1+5+i;
		}


		for (int i=0; i < 5; i++) {
			reguly->gray_ograniczenie_stawki[i].pozycja_startowa = przesuniecie + 1;
			reguly->gray_ograniczenie_stawki[i].dlugosc = 4;
			przesuniecie += 1+4;
			reguly->gray_ograniczenie_stawki_stawka[i].pozycja_startowa = przesuniecie;
			reguly->gray_ograniczenie_stawki_stawka[i].dlugosc = 8;
			przesuniecie +=8;
		}

	};


	void getCzyGracR1PTR(int przesuniecie, CzyGracR1 *reguly) {

		reguly->gray_wymaganych_glosow.pozycja_startowa=przesuniecie;
		reguly->gray_wymaganych_glosow.dlugosc=6;
		przesuniecie += 6;


		reguly->gray_para_w_rece.pozycja_startowa=przesuniecie+1;
		reguly->gray_para_w_rece.dlugosc=4;
		przesuniecie += 4 +1;
		reguly->gray_kolor_w_rece.pozycja_startowa=przesuniecie+1;
		reguly->gray_kolor_w_rece.dlugosc=4;
		przesuniecie += 4 +1;

		reguly->gray_ograniczenie_stawki.pozycja_startowa=przesuniecie+1;
		reguly->gray_ograniczenie_stawki.dlugosc=4;
		przesuniecie += 4 +1;
		reguly->gray_ograniczenie_stawki_wielkosc.pozycja_startowa=przesuniecie;
		reguly->gray_ograniczenie_stawki_wielkosc.dlugosc=10;
		przesuniecie += 10;

		reguly->gray_wysoka_karta_w_rece.pozycja_startowa=przesuniecie+1;
		reguly->gray_wysoka_karta_w_rece.dlugosc=4;
		przesuniecie += 4 +1;
		reguly->gray_bardzo_wysoka_karta_w_rece.pozycja_startowa=przesuniecie+1;
		reguly->gray_bardzo_wysoka_karta_w_rece.dlugosc=4;
		przesuniecie += 4 +1;

	};


}

#endif
