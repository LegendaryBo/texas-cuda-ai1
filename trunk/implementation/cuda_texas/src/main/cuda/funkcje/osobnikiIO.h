#include <iostream>
using std::cerr;
using std::cout;
using std::endl;
#include <fstream>
using std::ifstream;
#include <cstdlib>
using namespace std;
#include <sstream>

#include <stdio.h>
#include <stdlib.h>

#include "../struktury/osobnik.h"

#ifndef OSOBNIK_IO_H
#define OSOBNIK_IO_H


/**
 * ilePlikow - ile plikow bierze udzial
 * pKatalogZrodlowy - katalog zrodlowy z binarnymi osobnikami
 * rozmiarGenow - liczba intow, jaka przypada na kodon jednego osobnika
 */
ZbiorOsobnikow *odczytajOsobnikiZPliku(char *pPlikZrodlowy, int rozmiarGenow) {

	ifstream indata; // indata is like cin
	int liczbaOsobnikow;

//	printf("%s", pPlikZrodlowy);

	indata.open(pPlikZrodlowy); // opens the file
	if (!indata) { // file couldn't be opened
		cerr << "Error: nie mozna otworzyc pliku: \t " << pPlikZrodlowy << endl;
		exit(1);
	}

	liczbaOsobnikow = indata.get()*256*256*256 + indata.get()*256*256 + indata.get()*256 + indata.get();


	ZbiorOsobnikow *zbior_osobnikow = new ZbiorOsobnikow();


	Osobnik** osobnik = (Osobnik**)malloc( sizeof(Osobnik*) * liczbaOsobnikow );

	zbior_osobnikow->osobniki = osobnik;
	for (int i=0; i < liczbaOsobnikow; i++) {
		osobnik[i] = new Osobnik();
	}

	zbior_osobnikow->liczba_osobnikow=liczbaOsobnikow;

	for (int i=0; i < liczbaOsobnikow; i++) {
		osobnik[i]->geny = (int*)malloc(sizeof(int) * rozmiarGenow);
		for (int j=0; j < rozmiarGenow; j++)
			osobnik[i]->geny[j] = indata.get()*256*256*256 + indata.get()*256*256 + indata.get()*256 + indata.get();
	}
	indata.close();

	return zbior_osobnikow;
}

// funkcja odczytuje wszystkie osobniko z katalogow
// pKatalogZrodlowy/generacja1.bin
//  ..
// pKatalogZrodlowy/generacjaILE_PLIKOW.bin
ZbiorOsobnikow *odczytajOsobnikiZKatalogu(int ilePlikow,string pKatalogZrodlowy, int rozmiarGenow) {

	ZbiorOsobnikow **zbiory_osobnikow = (ZbiorOsobnikow**)malloc( sizeof(ZbiorOsobnikow*) * ilePlikow );

	int pSumarycznaLiczbaOsobnikow=0;
	for (int i=1; i <= ilePlikow; i++) {
		stringstream nazwaKatalogu;
		nazwaKatalogu << pKatalogZrodlowy << "generacja" << i << ".bin";
		zbiory_osobnikow[i-1] = odczytajOsobnikiZPliku( (char*)nazwaKatalogu.str().c_str(), rozmiarGenow );
		pSumarycznaLiczbaOsobnikow += zbiory_osobnikow[i-1]->liczba_osobnikow;
	}

	Osobnik **osobnik = (Osobnik**)malloc( sizeof(Osobnik*) * pSumarycznaLiczbaOsobnikow );
	int k=0;
	for (int i=0; i < ilePlikow; i++) {
		for (int j=0; j < zbiory_osobnikow[i]->liczba_osobnikow; j++) {
			osobnik[k] = zbiory_osobnikow[i]->osobniki[j] ;
			k++;
		}
	}
//	printf("kkkkkk %d",k);
//	printf("sumaryczna liczba osobnikow %d",pSumarycznaLiczbaOsobnikow);
	ZbiorOsobnikow *wszystkie_osobniki = new ZbiorOsobnikow;
	wszystkie_osobniki->osobniki=osobnik;
	wszystkie_osobniki->liczba_osobnikow=pSumarycznaLiczbaOsobnikow;
	return  wszystkie_osobniki;
}


#endif

