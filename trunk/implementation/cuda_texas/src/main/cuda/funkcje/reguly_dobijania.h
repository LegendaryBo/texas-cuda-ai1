#include "../struktury/texas_struktury.h"
#include "../struktury/rozdanie.h"
#include "../struktury/gra.h"
#include "../struktury/kod_graya.h"
#include "../struktury/dobijanie_r1.h"
#include "../struktury/dobijanie_rx.h"

#include "osobnik.h"
#include "rezultaty.h"

#include <stdio.h>


#ifndef REGULY_DOBIJANIA_H
#define REGULY_DOBIJANIA_H

extern "C" {


	__host__ void dobijajZawszeHOST(Gra *gra, int ktoryGracz, int gen_startowy, float *wynik)  {
		wynik[0] = getBitHOST(gra->gracze[ktoryGracz].geny, gen_startowy) * 1.0;

	};
	__device__ void dobijajZawszeDEVICE(Gra *gra, int ktoryGracz, int gen_startowy, float *wynik)  {
		wynik[0] = getBitDEVICE(gra->gracze[ktoryGracz].geny, gen_startowy) * 1.0;

	};

	__host__ void dobijajGdyParaWRekuR1HOST(Gra *gra, int ktoryGracz, int pozycja_genu, float *wynik, float stawka, float wspolczynnikDobijania) {
		wynik[0] =  (gra->rozdanie.karty_prywatne[ ktoryGracz ][0].wysokosc==gra->rozdanie.karty_prywatne[ ktoryGracz ][1].wysokosc) 
			  * ( getBitHOST(gra->gracze[ktoryGracz].geny, pozycja_genu) == 1 && stawka < gra->stawka * wspolczynnikDobijania)
			  * 1.0;
	};
	__device__ void dobijajGdyParaWRekuR1DEVICE(Gra *gra, int ktoryGracz, int pozycja_genu, float *wynik, float stawka, float wspolczynnikDobijania) {
		wynik[0] =  (gra->rozdanie.karty_prywatne[ ktoryGracz ][0].wysokosc==gra->rozdanie.karty_prywatne[ ktoryGracz ][1].wysokosc) 
			  * ( getBitDEVICE(gra->gracze[ktoryGracz].geny, pozycja_genu) == 1 && stawka < gra->stawka * wspolczynnikDobijania)
			  * 1.0;
	};


	__host__ void dobijajGdyWysokaKartaR1HOST(Gra *gra, int ktoryGracz, int pozycja_genu, float *wynik, float stawka, float wspolczynnikDobijania) {
		wynik[0] = (gra->rozdanie.karty_prywatne[ ktoryGracz ][0].wysokosc > 10 && gra->rozdanie.karty_prywatne[ ktoryGracz ][1].wysokosc > 10)
			  * ( getBitHOST(gra->gracze[ktoryGracz].geny, pozycja_genu) == 1 &&  stawka < gra->stawka * wspolczynnikDobijania)
			  * 1.0;
	};
	__device__ void dobijajGdyWysokaKartaR1DEVICE(Gra *gra, int ktoryGracz, int pozycja_genu, float *wynik, float stawka, float wspolczynnikDobijania) {
		wynik[0] = (gra->rozdanie.karty_prywatne[ ktoryGracz ][0].wysokosc > 10 && gra->rozdanie.karty_prywatne[ ktoryGracz ][1].wysokosc > 10)
			  * ( getBitDEVICE(gra->gracze[ktoryGracz].geny, pozycja_genu) == 1 &&  stawka < gra->stawka * wspolczynnikDobijania)
			  * 1.0;
	};	




	__host__ void dobijajGdyBrakujeXRXHOST(Gra *gra, int ktoryGracz, int pozycja_genu, float *wynik, float stawka, float wspolczynnikDobijania) {
		wynik[0] = ( getBitHOST(gra->gracze[ktoryGracz].geny, pozycja_genu) == 1 &&  stawka <= gra->stawka * wspolczynnikDobijania ) * 1.0;
	};
	__device__ void dobijajGdyBrakujeXRXDEVICE(Gra *gra, int ktoryGracz, int pozycja_genu, float *wynik, float stawka, float wspolczynnikDobijania) {
		wynik[0] = ( getBitDEVICE(gra->gracze[ktoryGracz].geny, pozycja_genu) == 1 &&  stawka <= gra->stawka * wspolczynnikDobijania ) * 1.0;
	};

	__host__ void dobijajGdyWysokaKartaRXHOST(Gra *gra, int ktoryGracz,  int pozycja_genu,  float *wynik, float stawka, int wymagany_rezultat) {
	    int rezultat_gracza = najlepszaKartaHOST( &gra->rozdanie.handy[ktoryGracz] );
	    wynik[0] = (rezultat_gracza >= wymagany_rezultat) 
		      * ( getBitHOST(gra->gracze[ktoryGracz].geny, pozycja_genu) == 1 && rezultat_gracza >= wymagany_rezultat) 
		      * 1.0;
	};
	__device__ void dobijajGdyWysokaKartaRXDEVICE(Gra *gra, int ktoryGracz,  int pozycja_genu,  float *wynik, float stawka, int wymagany_rezultat) {
	    int rezultat_gracza = najlepszaKartaDEVICE( &gra->rozdanie.handy[ktoryGracz] );
	    wynik[0] = (rezultat_gracza >= wymagany_rezultat) 
		      * ( getBitDEVICE(gra->gracze[ktoryGracz].geny, pozycja_genu) == 1 && rezultat_gracza >= wymagany_rezultat) 
		      * 1.0;
	};





	__host__ void aplikujDobijanieR1HOST(DobijanieR1 *reguly, Gra *gra, int ktoryGracz, float *output, float stawka) {

		float wyniki_reguly[2];

		output[0]=1.0;

		// regula 1
		dobijajZawszeHOST(gra, ktoryGracz, reguly->dobijac_zawsze, wyniki_reguly) ;
		if ( wyniki_reguly[0] == 1.0 ) {
			return;
		}

		float wspolczynnik[5];
		wspolczynnik[0]=0.3;
		wspolczynnik[1]=0.6;
		wspolczynnik[2]=0.8;
		wspolczynnik[3]=0.1;
		wspolczynnik[4]=0.3;

		// regula 2
		for (int i=0; i < 5; i++) {
			dobijajGdyBrakujeXRXHOST(gra, ktoryGracz, reguly->dobijac_brakuje_x[i], wyniki_reguly,  stawka, wspolczynnik[i] );
			if ( wyniki_reguly[0] == 1.0 ) {
				return;
			}
		}

		// regula 3
		dobijajGdyParaWRekuR1HOST(gra, ktoryGracz, reguly->dobijac_para_w_rece_01, wyniki_reguly,  stawka, 0.1);
		if ( wyniki_reguly[0] == 1.0 ) {
			return;
		}

		dobijajGdyParaWRekuR1HOST(gra, ktoryGracz, reguly->dobijac_para_w_rece_03, wyniki_reguly,  stawka, 0.3);
		if ( wyniki_reguly[0] == 1.0 ) {
			return;
		}

		dobijajGdyParaWRekuR1HOST(gra, ktoryGracz, reguly->dobijac_para_w_rece_06, wyniki_reguly,  stawka, 0.6);
		if ( wyniki_reguly[0] == 1.0 ) {
			return;
		}



		// regula 4
		dobijajGdyWysokaKartaR1HOST(gra, ktoryGracz, reguly->dobijac_wysoka_reka_01, wyniki_reguly,  stawka, 0.1);
		if ( wyniki_reguly[0] == 1.0 ) {
			return;
		}

		dobijajGdyWysokaKartaR1HOST(gra, ktoryGracz, reguly->dobijac_wysoka_reka_03, wyniki_reguly,  stawka, 0.3);
		if ( wyniki_reguly[0] == 1.0 ) {
			return;
		}

		dobijajGdyWysokaKartaR1HOST(gra, ktoryGracz, reguly->dobijac_wysoka_reka_06, wyniki_reguly,  stawka, 0.6);
		if ( wyniki_reguly[0] == 1.0 ) {
			return;
		}

		output[0] = 0.0;
		return;
	};
	__device__ void aplikujDobijanieR1DEVICE(DobijanieR1 *reguly, Gra *gra, int ktoryGracz, float *output, float stawka) {

		float wyniki_reguly[2];

		output[0]=1.0;

		// regula 1
		dobijajZawszeDEVICE(gra, ktoryGracz, reguly->dobijac_zawsze, wyniki_reguly) ;
		if ( wyniki_reguly[0] == 1.0 ) {
			return;
		}

		float wspolczynnik[5];
		wspolczynnik[0]=0.3;
		wspolczynnik[1]=0.6;
		wspolczynnik[2]=0.8;
		wspolczynnik[3]=0.1;
		wspolczynnik[4]=0.3;

		// regula 2
		for (int i=0; i < 5; i++) {
			dobijajGdyBrakujeXRXDEVICE(gra, ktoryGracz, reguly->dobijac_brakuje_x[i], wyniki_reguly,  stawka, wspolczynnik[i] );
			if ( wyniki_reguly[0] == 1.0 ) {
				return;
			}
		}

		// regula 3
		dobijajGdyParaWRekuR1DEVICE(gra, ktoryGracz, reguly->dobijac_para_w_rece_01, wyniki_reguly,  stawka, 0.1);
		if ( wyniki_reguly[0] == 1.0 ) {
			return;
		}

		dobijajGdyParaWRekuR1DEVICE(gra, ktoryGracz, reguly->dobijac_para_w_rece_03, wyniki_reguly,  stawka, 0.3);
		if ( wyniki_reguly[0] == 1.0 ) {
			return;
		}

		dobijajGdyParaWRekuR1DEVICE(gra, ktoryGracz, reguly->dobijac_para_w_rece_06, wyniki_reguly,  stawka, 0.6);
		if ( wyniki_reguly[0] == 1.0 ) {
			return;
		}



		// regula 4
		dobijajGdyWysokaKartaR1DEVICE(gra, ktoryGracz, reguly->dobijac_wysoka_reka_01, wyniki_reguly,  stawka, 0.1);
		if ( wyniki_reguly[0] == 1.0 ) {
			return;
		}

		dobijajGdyWysokaKartaR1DEVICE(gra, ktoryGracz, reguly->dobijac_wysoka_reka_03, wyniki_reguly,  stawka, 0.3);
		if ( wyniki_reguly[0] == 1.0 ) {
			return;
		}

		dobijajGdyWysokaKartaR1DEVICE(gra, ktoryGracz, reguly->dobijac_wysoka_reka_06, wyniki_reguly,  stawka, 0.6);
		if ( wyniki_reguly[0] == 1.0 ) {
			return;
		}

		output[0] = 0.0;
		return;
	};
	
	
	
	//
	/* */

	__host__ void aplikujDobijanieRXHOST(DobijanieRX *reguly, Gra *gra, int ktoryGracz, float *output, float stawka) {

		float wyniki_reguly[2];

		output[0]=1.0;

		float wspolczynnik[5];
		wspolczynnik[0]=0.3;
		wspolczynnik[1]=0.6;
		wspolczynnik[2]=0.8;
		wspolczynnik[3]=0.1;
		wspolczynnik[4]=0.3;

		for (int i=0; i < 5; i++) {

			dobijajGdyBrakujeXRXHOST(gra, ktoryGracz, reguly->poczatek+i, wyniki_reguly,  stawka, wspolczynnik[i]);
			if ( wyniki_reguly[0] == 1.0 ) {
				return;
			}
		}


		for (int i=2; i < 9; i++) {

			dobijajGdyWysokaKartaRXHOST(gra, ktoryGracz, reguly->poczatek+5+i-2, wyniki_reguly,  stawka, i);

				if ( wyniki_reguly[0] == 1.0 ) {
					return;
				}

		}

		output[0]=0.0;

	};
	__device__ void aplikujDobijanieRXDEVICE(DobijanieRX *reguly, Gra *gra, int ktoryGracz, float *output, float stawka) {

		float wyniki_reguly[2];

		output[0]=1.0;

		float wspolczynnik[5];
		wspolczynnik[0]=0.3;
		wspolczynnik[1]=0.6;
		wspolczynnik[2]=0.8;
		wspolczynnik[3]=0.1;
		wspolczynnik[4]=0.3;

		for (int i=0; i < 5; i++) {

			dobijajGdyBrakujeXRXDEVICE(gra, ktoryGracz, reguly->poczatek+i, wyniki_reguly,  stawka, wspolczynnik[i]);
			if ( wyniki_reguly[0] == 1.0 ) {
				return;
			}
		}


		for (int i=2; i < 9; i++) {

			dobijajGdyWysokaKartaRXDEVICE(gra, ktoryGracz, reguly->poczatek+5+i-2, wyniki_reguly,  stawka, i);

				if ( wyniki_reguly[0] == 1.0 ) {
					return;
				}

		}

		output[0]=0.0;

	};





	__host__ void getDobijanieR1PTR(int przesuniecie, DobijanieR1 *reguly) {

		reguly->dobijac_zawsze = przesuniecie;
		reguly->dobijac_brakuje_x[0] = przesuniecie+1;
		reguly->dobijac_brakuje_x[1] = przesuniecie+2;
		reguly->dobijac_brakuje_x[2] = przesuniecie+3;
		reguly->dobijac_brakuje_x[3] = przesuniecie+4;
		reguly->dobijac_brakuje_x[4] = przesuniecie+5;

		reguly->dobijac_para_w_rece_01 = przesuniecie+6;
		reguly->dobijac_para_w_rece_03 = przesuniecie+7;
		reguly->dobijac_para_w_rece_06 = przesuniecie+8;

		reguly->dobijac_wysoka_reka_01 = przesuniecie+9;
		reguly->dobijac_wysoka_reka_03 = przesuniecie+10;
		reguly->dobijac_wysoka_reka_06 = przesuniecie+11;

	};






	__host__ void getDobijanieRXPTR(int przesuniecie, DobijanieRX *dobijanie) {

		dobijanie->poczatek = przesuniecie;

	};


}

#endif
