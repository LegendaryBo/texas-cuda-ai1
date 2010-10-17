#include "../struktury/gra.h"
#include "../struktury/reguly.h"
#include "../struktury/kod_graya.h"
#include "../struktury/ile_grac_r1.h"
#include "../struktury/ile_grac_rx.h"
#include "../struktury/dobijanie_r1.h"
#include "../struktury/dobijanie_rx.h"
#include "../struktury/stawka_r1.h"
#include "../struktury/stawka_rx.h"
#include "../struktury/czy_grac_r1.h"
#include "../struktury/czy_grac_rx.h"


#include "../funkcje/generatory.h"

#include "../funkcje/reguly_ilegrac.h"
#include "../funkcje/reguly_stawka.h"
#include "../funkcje/reguly_dobijania.h"
#include "../funkcje/reguly_czygrac.h"
#include "../funkcje/gracz_AI.h"

#include <stdio.h>

#ifndef ROZGRYWKA_H
#define ROZGRYWKA_H

extern "C" {

	Gra *getGraPTR();


	void getBilans(Gra *gra,  float *output );

	void rozegrajNGier(int ktory_nasz, int **osobniki, float *wynik, int N, int liczba_intow, int liczba_osobnikow);

	void rozegrajNGierCUDA(int ktory_nasz, int **osobniki, float
*wynik, int N, int liczba_intow, int liczba_watkow, int liczba_osobnikow);

	void rozegrajNGierCUDAwithSeed(int ktory_nasz, int **osobniki, float
*wynik, int N, int liczba_intow, int liczba_watkow, int liczba_osobnikow, int initNumber);

	int **getIndividualPTRPTR(int size);

	void setIndividualPTR(int *IN, int **IN, int index);

	void destruktorGra(Gra *gra);

	void destruktorInt(int *ptr);

	void destruktorKodGraya(KodGraya *kodGraya);

	void destruktorRozdanie(Rozdanie *rozdanie);

	void destruktorHand(Hand *hand);


// rozpatrywanie to tablica zlozona z 6 elementow, w ktorej wartosciami sa 0  lub 1
__device__ __host__  int wygrany(Rozdanie *rozdanie, int *spasowani, int *wygrani) {

	int ilu_wygranych = 0;

	for (int i = 0; i < 6; i++) {
		if (spasowani[i] == 0) {
			if (ilu_wygranych == 0) {
				wygrani[0] = i;
				ilu_wygranych = 1;
			} else {
				int wynik = porownaj(rozdanie, i, wygrani[0]);

				if (wynik > 0) {
					wygrani[0] = i;
					ilu_wygranych = 1;
				}
				if (wynik == 0) {
					wygrani[ilu_wygranych] = i;
					ilu_wygranych++;
				}
			}
		}
	}

	return ilu_wygranych;
}


	__device__ __host__  void nowaGra(int *gracz1_geny, int *gracz2_geny, int *gracz3_geny,
			int *gracz4_geny, int *gracz5_geny, int *gracz6_geny,
			int nr_rozdania, int mode, Gra *gra) {


		gra->mode = mode;
		gra->gracze[0].geny = gracz1_geny;
		gra->gracze[1].geny = gracz2_geny;
		gra->gracze[2].geny = gracz3_geny;
		gra->gracze[3].geny = gracz4_geny;
		gra->gracze[4].geny = gracz5_geny;
		gra->gracze[5].geny = gracz6_geny;

		for (int i = 0; i < 6; i++) {
			gra->gracze[i].bilans = 0.0;
			gra->bids[i] = 0.0;
			gra->pass[i] = 0;
		}

		gra->minimal_bid = 10.0;;
		gra->stawka = gra->minimal_bid;
		gra->kto_na_musie = 4;
		gra->runda = 1;
		gra->pula = gra->minimal_bid + gra->minimal_bid / 2;
		gra->graczyWGrze = 6;
		int bla=nr_rozdania;
		generuj(nr_rozdania, &gra->rozdanie, &bla);



		gra->kto_podbil = -1;

	}
	;

	__device__ __host__ void wypiszKarte(Karta *karta) {

		if (karta->wysokosc >= 2 && karta->wysokosc <= 10)
			printf("%d ", karta->wysokosc);
		if ( karta->wysokosc == 11 )
			printf("walet ");
		if ( karta->wysokosc == 12 )
			printf("dama ");
		if ( karta->wysokosc == 13 )
			printf("krol ");
		if ( karta->wysokosc == 14 )
			printf("as ");

		if (karta->kolor == 1)
			printf("pik");
		if (karta->kolor == 2)
			printf("trefl");
		if (karta->kolor == 3)
			printf("kier");
		if (karta->kolor == 4)
			printf("karo");

	};


	__device__ __host__ void wypiszRozdanie(Rozdanie *rozdanie) {
		printf("publiczne karty: ");
		for (int i=0; i < 5; i++) {
			wypiszKarte( &rozdanie->karty_publiczne[i] );
			printf("  ");
		}
		printf("\n");

		for (int i=0; i < 6; i++) {
			printf("prywatne karty gracza nr %d : ", i);
			wypiszKarte( &rozdanie->karty_prywatne[i][0] );
			printf("  ");
			wypiszKarte( &rozdanie->karty_prywatne[i][1] );
			printf(" \n");
		}
	}



	Gra *getGraPtr() {
		Gra *gra = new Gra();
		return gra;
	}

	__device__ __host__ int rundaX_czy_grac(int runda) {
		return 1;
	}

	__device__ __host__ float rundaX_stawka(int raruna) {
		return 10.0;
	}

	__device__ __host__ int rundaX_dobijanie(float stawka, int runda) {
		return 1;
	}

	__device__ __host__ int rundaX_ileGrac(float stawka, Gra *gra, int i) {
		return 10;
	}





	__host__ float rundaX_czy_grac_ai3HOST(Gra *gra, int runda, int ktoryGracz, Reguly *reguly) {

		float output[1];

		if (runda==1)
			aplikujCzyGracR1HOST(&reguly->czy_grac_r1, gra, ktoryGracz, &output[0], 0.0);
		if (runda>1)
			aplikujCzyGracRXHOST(&reguly->czy_grac_rx[runda-2], gra, ktoryGracz, &output[0], 0.0);

		return output[0];
	}
	__device__ float rundaX_czy_grac_ai3DEVICE(Gra *gra, int runda, int ktoryGracz, Reguly *reguly) {

		float output[1];

		if (runda==1)
			aplikujCzyGracR1DEVICE(&reguly->czy_grac_r1, gra, ktoryGracz, &output[0], 0.0);
		if (runda>1)
			aplikujCzyGracRXDEVICE(&reguly->czy_grac_rx[runda-2], gra, ktoryGracz, &output[0], 0.0);

		return output[0];
	}


	__host__ float rundaX_stawka_ai3HOST(Gra *gra, int runda,  int ktoryGracz, Reguly *reguly) {

		float output[1];

		if (runda==1)
			aplikujStawkaR1HOST(&reguly->stawka_r1, gra, ktoryGracz, &output[0]) ;
		if (runda>1)
			aplikujStawkaRXHOST(&reguly->stawka_rx[runda-2], gra, ktoryGracz, &output[0]) ;
		return output[0];
	}
	__device__ float rundaX_stawka_ai3DEVICE(Gra *gra, int runda,  int ktoryGracz, Reguly *reguly) {

		float output[1];

		if (runda==1)
			aplikujStawkaR1DEVICE(&reguly->stawka_r1, gra, ktoryGracz, &output[0]) ;
		if (runda>1)
			aplikujStawkaRXDEVICE(&reguly->stawka_rx[runda-2], gra, ktoryGracz, &output[0]) ;
		return output[0];
	}


	__host__ float rundaX_dobijanie_ai3HOST(float stawka,Gra *gra, int runda,  int ktoryGracz, Reguly *reguly) {

		float output[1];

		if (runda==1)
			aplikujDobijanieR1HOST(&reguly->dobijanie_r1, gra, ktoryGracz, &output[0], stawka);
		if (runda>1)
			aplikujDobijanieRXHOST(&reguly->dobijanie_rx[runda-2], gra, ktoryGracz, &output[0], stawka);
		return output[0];
	}
	__device__ float rundaX_dobijanie_ai3DEVICE(float stawka,Gra *gra, int runda,  int ktoryGracz, Reguly *reguly) {

		float output[1];

		if (runda==1)
			aplikujDobijanieR1DEVICE(&reguly->dobijanie_r1, gra, ktoryGracz, &output[0], stawka);
		if (runda>1)
			aplikujDobijanieRXDEVICE(&reguly->dobijanie_rx[runda-2], gra, ktoryGracz, &output[0], stawka);
		return output[0];
	}

	__host__ float rundaX_ileGrac_ai3HOST(float stawka, Gra *gra, int runda, int ktoryGracz, Reguly *reguly) {

		float output[1];

		if (runda==1)
			aplikujIleGracR1HOST( &reguly->ile_grac_r1 , gra, ktoryGracz, &output[0], stawka);
		if (runda>1)
			aplikujIleGracRXHOST( &reguly->ile_grac_rx[runda-2] , gra, ktoryGracz, &output[0], stawka);

		return output[0];
	};
	__device__ float rundaX_ileGrac_ai3DEVICE(float stawka, Gra *gra, int runda, int ktoryGracz, Reguly *reguly) {

		float output[1];

		if (runda==1)
			aplikujIleGracR1DEVICE( &reguly->ile_grac_r1 , gra, ktoryGracz, &output[0], stawka);
		if (runda>1)
			aplikujIleGracRXDEVICE( &reguly->ile_grac_rx[runda-2] , gra, ktoryGracz, &output[0], stawka);

		return output[0];
	};





	__host__ float rundaX_czy_grac_ai3_r1HOST(Gra *gra, int runda, int ktoryGracz, Reguly *reguly) {
		float output[1];
		aplikujCzyGracR1HOST(&reguly->czy_grac_r1, gra, ktoryGracz, &output[0], 0.0);
		return output[0];
	}
	__device__ float rundaX_czy_grac_ai3_r1DEVICE(Gra *gra, int runda, int ktoryGracz, Reguly *reguly) {
		float output[1];
		aplikujCzyGracR1DEVICE(&reguly->czy_grac_r1, gra, ktoryGracz, &output[0], 0.0);
		return output[0];
	}





	__host__ float rundaX_stawka_ai3_r1HOST(Gra *gra, int runda,  int ktoryGracz, Reguly *reguly) {
		float output[1];
		aplikujStawkaR1HOST(&reguly->stawka_r1, gra, ktoryGracz, &output[0]) ;
		return output[0];
	}
	__device__ float rundaX_stawka_ai3_r1DEVICE(Gra *gra, int runda,  int ktoryGracz, Reguly *reguly) {
		float output[1];
		aplikujStawkaR1DEVICE(&reguly->stawka_r1, gra, ktoryGracz, &output[0]) ;
		return output[0];
	}




	__host__  float rundaX_dobijanie_ai3_r1HOST(float stawka,Gra *gra, int runda,  int ktoryGracz, Reguly *reguly) {
		float output[1];
		aplikujDobijanieR1HOST(&reguly->dobijanie_r1, gra, ktoryGracz, &output[0], stawka);
		return output[0];
	}
	__device__  float rundaX_dobijanie_ai3_r1DEVICE(float stawka,Gra *gra, int runda,  int ktoryGracz, Reguly *reguly) {
		float output[1];
		aplikujDobijanieR1DEVICE(&reguly->dobijanie_r1, gra, ktoryGracz, &output[0], stawka);
		return output[0];
	}


	__host__  float rundaX_ileGrac_ai3_r1HOST(float stawka, Gra *gra, int runda, int ktoryGracz, Reguly *reguly) {
		float output[1];
		aplikujIleGracR1HOST( &reguly->ile_grac_r1 , gra, ktoryGracz, &output[0], stawka);
		return output[0];
	};
	__device__  float rundaX_ileGrac_ai3_r1DEVICE(float stawka, Gra *gra, int runda, int ktoryGracz, Reguly *reguly) {
		float output[1];
		aplikujIleGracR1DEVICE( &reguly->ile_grac_r1 , gra, ktoryGracz, &output[0], stawka);
		return output[0];
	};


	__host__ float rundaX_czy_grac_ai3_rxHOST(Gra *gra, int runda, int ktoryGracz, Reguly *reguly) {
		float output[1];
		aplikujCzyGracRXHOST(&reguly->czy_grac_rx[runda-2], gra, ktoryGracz, &output[0], 0.0);
		return output[0];
	}
	__device__ float rundaX_czy_grac_ai3_rxDEVICE(Gra *gra, int runda, int ktoryGracz, Reguly *reguly) {
		float output[1];
		aplikujCzyGracRXDEVICE(&reguly->czy_grac_rx[runda-2], gra, ktoryGracz, &output[0], 0.0);
		return output[0];
	}



	__host__ float rundaX_stawka_ai3_rxHOST(Gra *gra, int runda,  int ktoryGracz, Reguly *reguly) {
		float output[1];
		aplikujStawkaRXHOST(&reguly->stawka_rx[runda-2], gra, ktoryGracz, &output[0]) ;
		return output[0];
	}
	__device__ float rundaX_stawka_ai3_rxDEVICE(Gra *gra, int runda,  int ktoryGracz, Reguly *reguly) {
		float output[1];
		aplikujStawkaRXDEVICE(&reguly->stawka_rx[runda-2], gra, ktoryGracz, &output[0]) ;
		return output[0];
	}


	__host__ float rundaX_dobijanie_ai3_rxHOST(float stawka,Gra *gra, int runda,  int ktoryGracz, Reguly *reguly) {
		float output[1];
		aplikujDobijanieRXHOST(&reguly->dobijanie_rx[runda-2], gra, ktoryGracz, &output[0], stawka);
		return output[0];
	}
	__device__ float rundaX_dobijanie_ai3_rxDEVICE(float stawka,Gra *gra, int runda,  int ktoryGracz, Reguly *reguly) {
		float output[1];
		aplikujDobijanieRXDEVICE(&reguly->dobijanie_rx[runda-2], gra, ktoryGracz, &output[0], stawka);
		return output[0];
	}



	__host__ float rundaX_ileGrac_ai3_rxHOST(float stawka, Gra *gra, int runda, int ktoryGracz, Reguly *reguly) {
		float output[1];
		aplikujIleGracRXHOST( &reguly->ile_grac_rx[runda-2] , gra, ktoryGracz, &output[0], stawka);
		return output[0];
	};
	__device__ float rundaX_ileGrac_ai3_rxDEVICE(float stawka, Gra *gra, int runda, int ktoryGracz, Reguly *reguly) {
		float output[1];
		aplikujIleGracRXDEVICE( &reguly->ile_grac_rx[runda-2] , gra, ktoryGracz, &output[0], stawka);
		return output[0];
	};


	__host__ void graj1RundeHOST(Gra *gra, int indeks_gracza, float *wynik, Reguly *reguly) {
			int i = gra->runda;

			int test = rundaX_czy_grac_ai3_r1HOST(gra, i, indeks_gracza, reguly);

			if (test != 1.0) {
				wynik[0] = -1.0;
				return;
			} else {

				float stawka = rundaX_stawka_ai3_r1HOST(gra, i, indeks_gracza, reguly);

				if (stawka < gra->stawka) {
					if (rundaX_dobijanie_ai3_r1HOST(stawka,gra,i,indeks_gracza, reguly) == 1.0)
						stawka = gra->stawka;
					else {
						wynik[0] =  -1.0;
						return;
					}
				}

				float ile = rundaX_ileGrac_ai3_r1HOST(stawka, gra, i, indeks_gracza, reguly);

				gra->gracze[indeks_gracza].bilans -=  ile - gra->bids[indeks_gracza];

				wynik[0] =  ile;
				return;
			}
	}
	__device__ void graj1RundeDEVICE(Gra *gra, int indeks_gracza, float *wynik, Reguly *reguly) {
			int i = gra->runda;

			int test = rundaX_czy_grac_ai3_r1DEVICE(gra, i, indeks_gracza, reguly);

			if (test != 1.0) {
				wynik[0] = -1.0;
				return;
			} else {

				float stawka = rundaX_stawka_ai3_r1DEVICE(gra, i, indeks_gracza, reguly);

				if (stawka < gra->stawka) {
					if (rundaX_dobijanie_ai3_r1DEVICE(stawka,gra,i,indeks_gracza, reguly) == 1.0)
						stawka = gra->stawka;
					else {
						wynik[0] =  -1.0;
						return;
					}
				}

				float ile = rundaX_ileGrac_ai3_r1DEVICE(stawka, gra, i, indeks_gracza, reguly);

				gra->gracze[indeks_gracza].bilans -=  ile - gra->bids[indeks_gracza];

				wynik[0] =  ile;
				return;
			}
	}


	__host__ void grajXRundeHOST(Gra *gra, int indeks_gracza, float *wynik, Reguly *reguly) {
			int i = gra->runda;
			int test = rundaX_czy_grac_ai3_rxHOST(gra, i, indeks_gracza, reguly);

			if (test != 1.0) {
				wynik[0] = -1.0;
				return;
			} else {

				float stawka = rundaX_stawka_ai3_rxHOST(gra, i, indeks_gracza, reguly);
				if (stawka < gra->stawka) {

					if (rundaX_dobijanie_ai3_rxHOST(stawka,gra,i,indeks_gracza, reguly) == 1.0)
						stawka = gra->stawka;
					else {
						wynik[0] =  -1.0;
						return;
					}
				}
				float ile = rundaX_ileGrac_ai3_rxHOST(stawka, gra, i, indeks_gracza, reguly);
				gra->gracze[indeks_gracza].bilans -=  ile - gra->bids[indeks_gracza];

				wynik[0] =  ile;
				return;
			}
	}
	__device__ void grajXRundeDEVICE(Gra *gra, int indeks_gracza, float *wynik, Reguly *reguly) {
			int i = gra->runda;
			int test = rundaX_czy_grac_ai3_rxDEVICE(gra, i, indeks_gracza, reguly);

			if (test != 1.0) {
				wynik[0] = -1.0;
				return;
			} else {

				float stawka = rundaX_stawka_ai3_rxDEVICE(gra, i, indeks_gracza, reguly);
				if (stawka < gra->stawka) {

					if (rundaX_dobijanie_ai3_rxDEVICE(stawka,gra,i,indeks_gracza, reguly) == 1.0)
						stawka = gra->stawka;
					else {
						wynik[0] =  -1.0;
						return;
					}
				}
				float ile = rundaX_ileGrac_ai3_rxDEVICE(stawka, gra, i, indeks_gracza, reguly);
				gra->gracze[indeks_gracza].bilans -=  ile - gra->bids[indeks_gracza];

				wynik[0] =  ile;
				return;
			}
	}




	__host__ void grajHOST(Gra *gra, int indeks_gracza, float *wynik, Reguly *reguly) {

		int i = gra->runda;

		if (gra->mode == 3) {
			int test = rundaX_czy_grac_ai3HOST(gra, i, indeks_gracza, reguly);


			if (test != 1.0) {
				wynik[0] = -1.0;
				return;
			} else {

				float stawka = rundaX_stawka_ai3HOST(gra, i, indeks_gracza, reguly);
				if (stawka < gra->stawka) {

					if (rundaX_dobijanie_ai3HOST(stawka,gra,i,indeks_gracza, reguly) == 1.0)
						stawka = gra->stawka;
					else {
						wynik[0] =  -1.0;
						return;
					}
				}

				float ile = rundaX_ileGrac_ai3HOST(stawka, gra, i, indeks_gracza, reguly);
				gra->gracze[indeks_gracza].bilans -=  ile - gra->bids[indeks_gracza];

				wynik[0] =  ile;
				return;
			}
		}


		if (gra->mode == 2) {
/*
			if (rundaX_czy_gracHOST(i) != 0) {
				wynik[0] =   -1.0;
				return;
			} else {

				float stawka = rundaX_stawkaHOST(i);

				if (stawka < gra->stawka) {
					if (rundaX_dobijanieHOST(gra->stawka, i) == 1)
						stawka = gra->stawka;
					else {
						wynik[0] =  -1.0;
						return;
					}
				}

				gra->gracze[indeks_gracza].bilans -= gra->stawka - gra->bids[indeks_gracza];


				float ile = rundaX_ileGracHOST(gra->stawka, gra, i);

				gra->gracze[indeks_gracza].bilans -=  ile - gra->bids[indeks_gracza];

				wynik[0] =   ile;
				return;
			}
*/
		}



		if (gra->mode == 1) {

			int max = 0;
			for (int i = 0; i < 7; i++) {
				if (gra->rozdanie.handy[indeks_gracza].karty[i].wysokosc
						> max)
					max= gra->rozdanie.handy[indeks_gracza].karty[i].wysokosc;
			}
			if (max < 6) {
				wynik[0] =   -1;
				return;
			}

			else {
				if (gra->stawka > max * 10) {
					gra->gracze[indeks_gracza].bilans -= gra->stawka
							- gra->bids[indeks_gracza];
					wynik[0] = gra->stawka;
					return;
				}
				else {
					gra->gracze[indeks_gracza].bilans -= max * 10.0
							- gra->bids[indeks_gracza];
					wynik[0] = max * 10.0;
					return;
				}
			}

		}

		if (gra->mode == 0) {
			gra->gracze[indeks_gracza].bilans -= 20.0
					- gra->bids[indeks_gracza];
			wynik[0] =   20.0;
			return;
		}


	};


	__device__ void grajDEVICE(Gra *gra, int indeks_gracza, float *wynik, Reguly *reguly) {

		int i = gra->runda;

		if (gra->mode == 3) {
			int test = rundaX_czy_grac_ai3DEVICE(gra, i, indeks_gracza, reguly);


			if (test != 1.0) {
				wynik[0] = -1.0;
				return;
			} else {

				float stawka = rundaX_stawka_ai3DEVICE(gra, i, indeks_gracza, reguly);
				if (stawka < gra->stawka) {

					if (rundaX_dobijanie_ai3DEVICE(stawka,gra,i,indeks_gracza, reguly) == 1.0)
						stawka = gra->stawka;
					else {
						wynik[0] =  -1.0;
						return;
					}
				}

				float ile = rundaX_ileGrac_ai3DEVICE(stawka, gra, i, indeks_gracza, reguly);
				gra->gracze[indeks_gracza].bilans -=  ile - gra->bids[indeks_gracza];

				wynik[0] =  ile;
				return;
			}
		}


		if (gra->mode == 2) {
/*
			if (rundaX_czy_grac(i) != 0) {
				wynik[0] =   -1.0;
				return;
			} else {

				float stawka = rundaX_stawka(i);

				if (stawka < gra->stawka) {
					if (rundaX_dobijanieDEVICE(gra->stawka, i) == 1)
						stawka = gra->stawka;
					else {
						wynik[0] =  -1.0;
						return;
					}
				}

				gra->gracze[indeks_gracza].bilans -= gra->stawka - gra->bids[indeks_gracza];


				float ile = rundaX_ileGracDEVICE(gra->stawka, gra, i);

				gra->gracze[indeks_gracza].bilans -=  ile - gra->bids[indeks_gracza];

				wynik[0] =   ile;
				return;
			}
*/
		}



		if (gra->mode == 1) {

			int max = 0;
			for (int i = 0; i < 7; i++) {
				if (gra->rozdanie.handy[indeks_gracza].karty[i].wysokosc
						> max)
					max= gra->rozdanie.handy[indeks_gracza].karty[i].wysokosc;
			}
			if (max < 6) {
				wynik[0] =   -1;
				return;
			}

			else {
				if (gra->stawka > max * 10) {
					gra->gracze[indeks_gracza].bilans -= gra->stawka
							- gra->bids[indeks_gracza];
					wynik[0] = gra->stawka;
					return;
				}
				else {
					gra->gracze[indeks_gracza].bilans -= max * 10.0
							- gra->bids[indeks_gracza];
					wynik[0] = max * 10.0;
					return;
				}
			}

		}

		if (gra->mode == 0) {
			gra->gracze[indeks_gracza].bilans -= 20.0
					- gra->bids[indeks_gracza];
			wynik[0] =   20.0;
			return;
		}


	};





	__device__ __host__  void wszyscy_spasowali(Gra *gra) {

		//printf("wszyscy spasowali\n");
		for (int i = 0; i < 6; i++) {
			if (gra->pass[i] != 1)
				gra->gracze[i].bilans += gra->pula;
		}

	}




	__host__ int grajRundeHOST(Gra *gra, Reguly *reguly, int ktora_runda) {

		setRunda(gra, ktora_runda);

		gra->kto_podbil = 0;
		for (int i = 5; i > 0; i--)
			if (gra->pass[i] != 1) {
				gra->kto_podbil = i;
				break;
			}
		gra->stawka = 0;

		for (int i = 0; i < 6; i++)
			gra->bids[i] = 0;

		int i=gra->kto_podbil;
		int pierwszy_gracz=0;
		while (true) {

				if (gra->kto_podbil == i%6    && pierwszy_gracz==1) {

					if (ktora_runda == 4) {
						int wygrani[6];

						sprawdzRezultatyHOST(&gra->rozdanie);

						int ilu_wygralo = wygrany(&gra->rozdanie,
								gra->pass, &wygrani[0]);

						for (int j = 0; j < ilu_wygralo; j++) {
		//					printf("gracz %d wygral \n",wygrani[j]);
							gra->gracze[wygrani[j]].bilans
									+= gra->pula
											/ ilu_wygralo;
						}
					}
					return 0;
				}
				pierwszy_gracz=1;


				if (gra->pass[i%6] == 1) {
	//				printf("runda3 gracz %d spasowal, olewam \n",i%6);
					i++;
					continue;
				}

				// gdy wszyscy oprocz jednej osoby spasowali
				if (gra->graczyWGrze == 1) {
					wszyscy_spasowali(gra);
					return -10;
				}

				float bid;
				grajXRundeHOST(gra, i%6, &bid, reguly);



				if (bid < gra->stawka) {
					gra->pass[i%6] = 1;
					gra->graczyWGrze--;
				} else {

					if (bid > gra->stawka) {
						gra->kto_podbil = i%6;
						gra->stawka = bid;
					}
					if (bid >= gra->stawka) {
						gra->pula += gra->stawka - gra->bids[i%6];
						gra->bids[i%6] = bid;
					}

				}
			i++;
		}
	}
	__device__ int grajRundeDEVICE(Gra *gra, Reguly *reguly, int ktora_runda) {

		setRunda(gra, ktora_runda);

		gra->kto_podbil = 0;
		for (int i = 5; i > 0; i--)
			if (gra->pass[i] != 1) {
				gra->kto_podbil = i;
				break;
			}
		gra->stawka = 0;

		for (int i = 0; i < 6; i++)
			gra->bids[i] = 0;

		int i=gra->kto_podbil;
		int pierwszy_gracz=0;
		while (true) {

				if (gra->kto_podbil == i%6    && pierwszy_gracz==1) {

					if (ktora_runda == 4) {
						int wygrani[6];

						sprawdzRezultatyDEVICE(&gra->rozdanie);

						int ilu_wygralo = wygrany(&gra->rozdanie,
								gra->pass, &wygrani[0]);

						for (int j = 0; j < ilu_wygralo; j++) {
		//					printf("gracz %d wygral \n",wygrani[j]);
							gra->gracze[wygrani[j]].bilans
									+= gra->pula
											/ ilu_wygralo;
						}
					}
					return 0;
				}
				pierwszy_gracz=1;


				if (gra->pass[i%6] == 1) {
	//				printf("runda3 gracz %d spasowal, olewam \n",i%6);
					i++;
					continue;
				}

				// gdy wszyscy oprocz jednej osoby spasowali
				if (gra->graczyWGrze == 1) {
					wszyscy_spasowali(gra);
					return -10;
				}

				float bid;
				grajXRundeDEVICE(gra, i%6, &bid, reguly);



				if (bid < gra->stawka) {
					gra->pass[i%6] = 1;
					gra->graczyWGrze--;
				} else {

					if (bid > gra->stawka) {
						gra->kto_podbil = i%6;
						gra->stawka = bid;
					}
					if (bid >= gra->stawka) {
						gra->pula += gra->stawka - gra->bids[i%6];
						gra->bids[i%6] = bid;
					}

				}
			i++;
		}
	}





	__host__ float rozegrajPartieHOST(Gra *gra, int ktory_gracz, Reguly *reguly) {



		gra->gracze[gra->kto_na_musie % 6].bilans -= gra->minimal_bid;
		gra->bids[gra->kto_na_musie % 6] = gra->minimal_bid;

		gra->gracze[(gra->kto_na_musie + 1) % 6].bilans -= gra->minimal_bid / 2;
		gra->bids[(gra->kto_na_musie + 1) % 6] = gra->minimal_bid / 2;

		gra->pula = gra->minimal_bid + gra->minimal_bid / 2;
		gra->stawka = gra->minimal_bid;

		gra->kto_podbil = -1;

		while (1 == 1) {
			for (int i = 0; i < 6; i++) {

				if (gra->graczyWGrze == 1) {
					wszyscy_spasowali(gra);
					//printf("wszyscy spasowali \n");
					return gra->gracze[ktory_gracz].bilans;
				}

				if (gra->kto_podbil == i) {

					for (int k=2; k <= 4; k++)
						if ( grajRundeHOST(gra, reguly, k) == -10 )
							return gra->gracze[ktory_gracz].bilans;

	//				printf("koniec gry \n");
					return gra->gracze[ktory_gracz].bilans;
				}

				if (gra->kto_podbil == -1) {
					gra->kto_podbil = 0;
				}

				if (gra->pass[i] == 1)
					continue;

				if (gra->graczyWGrze == 1) {
					wszyscy_spasowali(gra);
					return gra->gracze[ktory_gracz].bilans;
				}

				float bid;
				graj1RundeHOST(gra, i, &bid, reguly);


				if (bid == -1.0) {
	//				printf("gracz nr %d spasowal \n",i);

					gra->pass[i] = 1;
					gra->graczyWGrze--;


				} else {
					if (bid > gra->stawka) {
						gra->kto_podbil = i;
						gra->stawka = bid;
					}
					if (bid >= gra->stawka) {
						gra->pula += gra->stawka - gra->bids[i];
						gra->bids[i] = bid;
					}
				}

			}

		}

	}
	;
	__device__ float rozegrajPartieDEVICE(Gra *gra, int ktory_gracz, Reguly *reguly) {



		gra->gracze[gra->kto_na_musie % 6].bilans -= gra->minimal_bid;
		gra->bids[gra->kto_na_musie % 6] = gra->minimal_bid;

		gra->gracze[(gra->kto_na_musie + 1) % 6].bilans -= gra->minimal_bid / 2;
		gra->bids[(gra->kto_na_musie + 1) % 6] = gra->minimal_bid / 2;

		gra->pula = gra->minimal_bid + gra->minimal_bid / 2;
		gra->stawka = gra->minimal_bid;

		gra->kto_podbil = -1;

		while (1 == 1) {
			for (int i = 0; i < 6; i++) {

				if (gra->graczyWGrze == 1) {
					wszyscy_spasowali(gra);
	//				printf("wszyscy spasowali \n");
					return gra->gracze[ktory_gracz].bilans;
				}

				if (gra->kto_podbil == i) {

					for (int k=2; k <= 4; k++)
						if ( grajRundeDEVICE(gra, reguly, k) == -10 )
							return gra->gracze[ktory_gracz].bilans;

	//				printf("koniec gry \n");
					return gra->gracze[ktory_gracz].bilans;
				}

				if (gra->kto_podbil == -1) {
					gra->kto_podbil = 0;
				}

				if (gra->pass[i] == 1)
					continue;

				if (gra->graczyWGrze == 1) {
					wszyscy_spasowali(gra);
					return gra->gracze[ktory_gracz].bilans;
				}

				float bid;
				graj1RundeDEVICE(gra, i, &bid, reguly);


				if (bid == -1.0) {
	//				printf("gracz nr %d spasowal \n",i);

					gra->pass[i] = 1;
					gra->graczyWGrze--;


				} else {
					if (bid > gra->stawka) {
						gra->kto_podbil = i;
						gra->stawka = bid;
					}
					if (bid >= gra->stawka) {
						gra->pula += gra->stawka - gra->bids[i];
						gra->bids[i] = bid;
					}
				}

			}

		}

	}
	;


}


#endif
