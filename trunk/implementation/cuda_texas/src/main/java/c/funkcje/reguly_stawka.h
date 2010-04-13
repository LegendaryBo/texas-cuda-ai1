#include "../struktury/texas_struktury.h"
#include "../struktury/rozdanie.h"
#include "../struktury/gra.h"
#include "../struktury/kod_graya.h"
#include "../struktury/stawka_r1.h"
#include "../struktury/stawka_rx.h"

#include "osobnik.h"
#include "rezultaty.h"


#ifndef REGULY_STAWKA_H
#define REGULY_STAWKA_H

extern "C" {

	__host__  void stawkaParaWRekuR1HOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya,  float *wynik) {
		wynik[0] = (gra->rozdanie.karty_prywatne[ ktoryGracz ][0].wysokosc==gra->rozdanie.karty_prywatne[ ktoryGracz ][1].wysokosc)
			    * ( getBitHOST(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1)
			    * obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya );
	};
	__device__  void stawkaParaWRekuR1DEVICE(Gra *gra, int ktoryGracz, KodGraya *kod_graya,  float *wynik) {
		wynik[0] = (gra->rozdanie.karty_prywatne[ ktoryGracz ][0].wysokosc==gra->rozdanie.karty_prywatne[ ktoryGracz ][1].wysokosc)
			    * ( getBitDEVICE(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1)
			    * obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya );
	};



	__host__ void stawkaKolorWRekuR1HOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya,  float *wynik) {
	      wynik[0] = (gra->rozdanie.karty_prywatne[ ktoryGracz ][0].kolor==gra->rozdanie.karty_prywatne[ ktoryGracz ][1].kolor)
			  * ( getBitHOST(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1)
			  *  obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya );
	};
	__device__ void stawkaKolorWRekuR1DEVICE(Gra *gra, int ktoryGracz, KodGraya *kod_graya,  float *wynik) {
	      wynik[0] = (gra->rozdanie.karty_prywatne[ ktoryGracz ][0].kolor==gra->rozdanie.karty_prywatne[ ktoryGracz ][1].kolor)
			  * ( getBitDEVICE(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1)
			  *  obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya );
	};


	__host__ void stawkaWysokaKartaWRekuR1HOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya,  float *wynik) {
		wynik[0] = (gra->rozdanie.karty_prywatne[ ktoryGracz ][0].wysokosc > 10&&gra->rozdanie.karty_prywatne[ ktoryGracz ][1].wysokosc  > 10)
			    * ( getBitHOST(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1)
			    * obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya );
	};
	__device__ void stawkaWysokaKartaWRekuR1DEVICE(Gra *gra, int ktoryGracz, KodGraya *kod_graya,  float *wynik) {
		wynik[0] = (gra->rozdanie.karty_prywatne[ ktoryGracz ][0].wysokosc > 10&&gra->rozdanie.karty_prywatne[ ktoryGracz ][1].wysokosc  > 10)
			    * ( getBitDEVICE(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1)
			    * obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya );
	};

	__host__ void stawkaBardzoWysokaKartaWRekuR1HOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya,  float *wynik) {
		wynik[0] = (gra->rozdanie.karty_prywatne[ ktoryGracz ][0].wysokosc >12&&gra->rozdanie.karty_prywatne[ ktoryGracz ][1].wysokosc  > 12)
			    * ( getBitHOST(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1)
			    * obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya );
	};
	__device__ void stawkaBardzoWysokaKartaWRekuR1DEVICE(Gra *gra, int ktoryGracz, KodGraya *kod_graya,  float *wynik) {
		wynik[0] = (gra->rozdanie.karty_prywatne[ ktoryGracz ][0].wysokosc >12&&gra->rozdanie.karty_prywatne[ ktoryGracz ][1].wysokosc  > 12)
			    * ( getBitDEVICE(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1)
			    * obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya );
	};

	__host__ void stawkaStalaHOST(Gra *gra, int ktoryGracz,KodGraya *kod_graya,  float *wynik) {
		wynik[0] = ( getBitHOST(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1) 
			    * obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, kod_graya ) ;
	};
	__device__ void stawkaStalaDEVICE(Gra *gra, int ktoryGracz,KodGraya *kod_graya,  float *wynik) {
		wynik[0] = ( getBitDEVICE(gra->gracze[ktoryGracz].geny, kod_graya->pozycja_startowa-1) == 1) 
			    * obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, kod_graya ) ;
	};

	void getStawkaR1PTR(int przesuniecie, StawkaR1 *stawka) {

		stawka->gray_para.pozycja_startowa = przesuniecie + 1;
		stawka->gray_para.dlugosc = 8;
		przesuniecie += 1+8;

		stawka->gray_kolor.pozycja_startowa = przesuniecie + 1;
		stawka->gray_kolor.dlugosc = 8;
		przesuniecie += 1+8;

		stawka->gray_kolor2.pozycja_startowa = przesuniecie + 1;
		stawka->gray_kolor2.dlugosc = 8;
		przesuniecie += 1+8;

		stawka->gray_wysokie_karty.pozycja_startowa = przesuniecie + 1;
		stawka->gray_wysokie_karty.dlugosc = 8;
		przesuniecie += 1+8;

		stawka->gray_bardzo_wysokie_karty.pozycja_startowa = przesuniecie + 1;
		stawka->gray_bardzo_wysokie_karty.dlugosc = 8;
		przesuniecie += 1+8;

		stawka->gray_stala_stawka_mala.pozycja_startowa = przesuniecie + 1;
		stawka->gray_stala_stawka_mala.dlugosc = 3;
		przesuniecie += 1+3;

		stawka->gray_stala_stawka_srednia.pozycja_startowa = przesuniecie + 1;
		stawka->gray_stala_stawka_srednia.dlugosc = 5;
		przesuniecie += 1+5;

		stawka->gray_stala_stawka_duza.pozycja_startowa = przesuniecie + 1;
		stawka->gray_stala_stawka_duza.dlugosc = 9;
		przesuniecie += 1+9;

	};



	__host__ void aplikujStawkaR1HOST(StawkaR1 *reguly, Gra *gra, int ktoryGracz, float *output) {

		float stawka = 0.0;
		float regula_wynik=0;

		stawkaParaWRekuR1HOST(gra, ktoryGracz, &reguly->gray_para, &regula_wynik);
		stawka += regula_wynik * gra->minimal_bid;

		stawkaKolorWRekuR1HOST(gra, ktoryGracz, &reguly->gray_kolor, &regula_wynik);
		stawka += regula_wynik * gra->minimal_bid;

		stawkaKolorWRekuR1HOST(gra, ktoryGracz, &reguly->gray_kolor2, &regula_wynik);
		stawka += regula_wynik * gra->minimal_bid;

		stawkaWysokaKartaWRekuR1HOST(gra, ktoryGracz, &reguly->gray_wysokie_karty, &regula_wynik);
		stawka += regula_wynik * gra->minimal_bid;

		stawkaBardzoWysokaKartaWRekuR1HOST(gra, ktoryGracz, &reguly->gray_bardzo_wysokie_karty, &regula_wynik);
		stawka += regula_wynik * gra->minimal_bid;

		stawkaStalaHOST(gra, ktoryGracz, &reguly->gray_stala_stawka_mala, &regula_wynik);
		stawka += regula_wynik * gra->minimal_bid;

		stawkaStalaHOST(gra, ktoryGracz, &reguly->gray_stala_stawka_srednia, &regula_wynik);
		stawka += regula_wynik * gra->minimal_bid;

		stawkaStalaHOST(gra, ktoryGracz, &reguly->gray_stala_stawka_duza, &regula_wynik);
		stawka += regula_wynik * gra->minimal_bid;

		output[0] = stawka;
	};
	__device__ void aplikujStawkaR1DEVICE(StawkaR1 *reguly, Gra *gra, int ktoryGracz, float *output) {

		float stawka = 0.0;
		float regula_wynik=0;

		stawkaParaWRekuR1DEVICE(gra, ktoryGracz, &reguly->gray_para, &regula_wynik);
		stawka += regula_wynik * gra->minimal_bid;

		stawkaKolorWRekuR1DEVICE(gra, ktoryGracz, &reguly->gray_kolor, &regula_wynik);
		stawka += regula_wynik * gra->minimal_bid;

		stawkaKolorWRekuR1DEVICE(gra, ktoryGracz, &reguly->gray_kolor2, &regula_wynik);
		stawka += regula_wynik * gra->minimal_bid;

		stawkaWysokaKartaWRekuR1DEVICE(gra, ktoryGracz, &reguly->gray_wysokie_karty, &regula_wynik);
		stawka += regula_wynik * gra->minimal_bid;

		stawkaBardzoWysokaKartaWRekuR1DEVICE(gra, ktoryGracz, &reguly->gray_bardzo_wysokie_karty, &regula_wynik);
		stawka += regula_wynik * gra->minimal_bid;

		stawkaStalaDEVICE(gra, ktoryGracz, &reguly->gray_stala_stawka_mala, &regula_wynik);
		stawka += regula_wynik * gra->minimal_bid;

		stawkaStalaDEVICE(gra, ktoryGracz, &reguly->gray_stala_stawka_srednia, &regula_wynik);
		stawka += regula_wynik * gra->minimal_bid;

		stawkaStalaDEVICE(gra, ktoryGracz, &reguly->gray_stala_stawka_duza, &regula_wynik);
		stawka += regula_wynik * gra->minimal_bid;

		output[0] = stawka;
	};







	void getStawkaRXPTR(int przesuniecie, StawkaRX *stawka) {

		for (int i=2; i < 9; i++) {
			stawka->gray_jest_rezultat[i-2].pozycja_startowa = przesuniecie + 1;
			stawka->gray_jest_rezultat[i-2].dlugosc =5+i;
			przesuniecie += 1+5+i;
		}
		for (int i=0; i < 5; i++) {
			stawka->gray_mala_stawka[i].pozycja_startowa = przesuniecie + 1;
			stawka->gray_mala_stawka[i].dlugosc = 4;
			przesuniecie += 1+4;
			stawka->gray_mala_stawka_parametr[i].pozycja_startowa = przesuniecie;
			stawka->gray_mala_stawka_parametr[i].dlugosc = 8;
			przesuniecie += 8;
		}

	};

	__host__ void stawkaWysokaKartaRXHOST(Gra *gra, int ktoryGracz,KodGraya *ileGrac,  float *wynik, int wymagany_rezultat) {
		int rezultat_gracza = najlepszaKartaHOST( &gra->rozdanie.handy[ktoryGracz] );

		wynik[0] = (rezultat_gracza >= wymagany_rezultat)
			    * ( getBitHOST(gra->gracze[ktoryGracz].geny, ileGrac->pozycja_startowa-1) == 1)
			    * (float)obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, ileGrac );
	};
	__device__ void stawkaWysokaKartaRXDEVICE(Gra *gra, int ktoryGracz,KodGraya *ileGrac,  float *wynik, int wymagany_rezultat) {
		int rezultat_gracza = najlepszaKartaDEVICE( &gra->rozdanie.handy[ktoryGracz] );

		wynik[0] = (rezultat_gracza >= wymagany_rezultat)
			    * ( getBitDEVICE(gra->gracze[ktoryGracz].geny, ileGrac->pozycja_startowa-1) == 1)
			    * (float)obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, ileGrac );
	};	
	
	

	__host__ void StawkaLicytujGdyMalaHOST(Gra *gra, int ktoryGracz,KodGraya *gray_stawka,KodGraya *limit_stawki,  float *wynik) {
	    wynik[0] = (gra->stawka<=obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, limit_stawki ) * gra->minimal_bid)
			* ( getBitHOST( gra->gracze[ktoryGracz].geny, gray_stawka->pozycja_startowa-1) == 1 )
			* (float)obliczKodGrayaHOST( gra->gracze[ktoryGracz].geny, gray_stawka );
	};
	__device__ void StawkaLicytujGdyMalaDEVICE(Gra *gra, int ktoryGracz,KodGraya *gray_stawka,KodGraya *limit_stawki,  float *wynik) {
	    wynik[0] = (gra->stawka<=obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, limit_stawki ) * gra->minimal_bid)
			* ( getBitDEVICE( gra->gracze[ktoryGracz].geny, gray_stawka->pozycja_startowa-1) == 1 )
			* (float)obliczKodGrayaDEVICE( gra->gracze[ktoryGracz].geny, gray_stawka );
	};



	__host__  void aplikujStawkaRXHOST(StawkaRX *reguly, Gra *gra, int ktoryGracz, float *output) {

		float stawka = 0.0;
		float regula_wynik=0;

		for (int i=2; i < 9; i++) {
			stawkaWysokaKartaRXHOST(gra, ktoryGracz,
					&reguly->gray_jest_rezultat[i-2],
					&regula_wynik, i);
			stawka += gra->minimal_bid * regula_wynik;
		}

		for (int i=0; i < 5; i++) {
			StawkaLicytujGdyMalaHOST(gra, ktoryGracz, &reguly->gray_mala_stawka[i],
					&reguly->gray_mala_stawka_parametr[i],  &regula_wynik);
			stawka += gra->minimal_bid *  regula_wynik;
		}
		output[0] = stawka;
	};
	__device__  void aplikujStawkaRXDEVICE(StawkaRX *reguly, Gra *gra, int ktoryGracz, float *output) {

		float stawka = 0.0;
		float regula_wynik=0;

		for (int i=2; i < 9; i++) {
			stawkaWysokaKartaRXDEVICE(gra, ktoryGracz,
					&reguly->gray_jest_rezultat[i-2],
					&regula_wynik, i);
			stawka += gra->minimal_bid * regula_wynik;
		}

		for (int i=0; i < 5; i++) {
			StawkaLicytujGdyMalaDEVICE(gra, ktoryGracz, &reguly->gray_mala_stawka[i],
					&reguly->gray_mala_stawka_parametr[i],  &regula_wynik);
			stawka += gra->minimal_bid *  regula_wynik;
		}
		output[0] = stawka;
	};


};

#endif
