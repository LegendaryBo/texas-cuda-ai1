%module ai_texas_swig
%{
#include "../../classes/struktury/texas_struktury.h"
#include "../../classes/struktury/gra.h"
#include "../../classes/struktury/gracz.h"
#include "../../classes/struktury/kod_graya.h"
#include "../../classes/struktury/ile_grac_r1.h"
#include "../../classes/struktury/stawka_r1.h"
#include "../../classes/struktury/dobijanie_r1.h"
#include "../../classes/struktury/czy_grac_r1.h"
#include "../../classes/struktury/ile_grac_rx.h"
#include "../../classes/struktury/stawka_rx.h"
#include "../../classes/struktury/dobijanie_rx.h"
#include "../../classes/struktury/czy_grac_rx.h"
#include "../../classes/struktury/reguly.h"

extern void destruktorGra(Gra *gra);
extern void destruktorInt(int *ptr);
extern void destruktorKodGraya(KodGraya *ptr);
extern void destruktorRozdanie(Rozdanie *ptr);
extern void destruktorHand(Hand *ptr);

extern void rozegrajNGier(int ktory_nasz, int **osobniki, float *wynik, int N, int liczba_intow, int liczba_osobnikow);
extern void rozegrajNGierCUDA(int ktory_nasz, int **osobniki, float *wynik, int N, int liczba_intow, int liczba_watkow, int liczba_osobnikow);
extern int **getIndividualPTRPTR(int size);
extern void setIndividualPTR(int *osobnik, int **pointer, int index);

extern int obliczKodGrayaHOST(int *, KodGraya *);

extern float rozegrajPartieHOST(Gra *gra, int ktory_gracz, Reguly *reguly);
extern KodGraya  *getKodGrayaPTR(int pozycja_startowa, int dlugosc);
extern int *getOsobnikPTR(int *geny, int dlugosc);
extern Gra *getGraPtr();

extern void grajHOST(Gra *gra, int indeks_gracza, float *wynik, Reguly *reguly);

extern void setIleGraczyWGrze(Gra *gra, int ile);
extern void setPula(Gra *gra, int ile);
extern void setStawka(Gra *gra, int ile);
extern void setRunda(Gra *gra, int ile);
extern void setMode(Gra *gra, int ile);

extern void ileGracParaWRekuR1HOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya, KodGraya *kod_graya_jak_grac,  float *wynik);
extern void ileGracKolorWRekuR1HOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya, KodGraya *kod_graya_jak_grac,  float *wynik);
extern void ileGracWysokaKartaWRekuR1HOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya, KodGraya *kod_graya_jak_grac,  float *wynik);
extern void ileGracBardzoWysokaKartaWRekuR1HOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya, KodGraya *kod_graya_jak_grac,  float *wynik);
extern void IleGracXGraczyWGrzeRXHOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya, KodGraya *kod_graya_jak_grac,  float *wynik, int ileGraczy);
extern void IleGracPulaRXHOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya, KodGraya *kod_graya_jak_grac, KodGraya *kod_graya_pula,  float *wynik);
extern void IleGracStawkaRXHOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya, KodGraya *kod_graya_jak_grac, KodGraya *kod_graya_pula,  float *wynik);
extern void IleGracRezultatRXHOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya, KodGraya *kod_graya_jak_grac,  float *wynik, int rezultat);

extern void dobijajZawszeHOST(Gra *gra, int ktoryGracz, int gen_startowy, float *wynik);
extern void dobijajGdyParaWRekuR1HOST(Gra *gra, int ktoryGracz,  int pozycja_genu, float *wynik, float stawka, float wspolczynnikDobijania);
extern void dobijajGdyWysokaKartaR1HOST(Gra *gra, int ktoryGracz,  int pozycja_genu, float *wynik, float stawka, float wspolczynnikDobijania);
extern void dobijajGdyBrakujeXRXHOST(Gra *gra, int ktoryGracz,  int pozycja_genu, float *wynik, float stawka, float wspolczynnikDobijania);
extern void dobijajGdyWysokaKartaRXHOST(Gra *gra, int ktoryGracz,  int pozycja_genu,  float *wynik, float stawka, int wymagany_rezultat);


extern void grajRezultatRXHOST(Gra *gra, int ktoryGracz,  KodGraya *kod_graya,  float *wynik, int wymagany_rezultat);
extern void ograniczenieStawkiRXHOST(Gra *gra, int ktoryGracz,  KodGraya *kod_graya, KodGraya *kod_graya2,  float *wynik);
extern void grajGdyParaWRekuR1HOST(Gra *gra, int ktoryGracz,KodGraya *kod_graya,  float *wynik, float stawka);
extern void grajGdyKolorWRekuR1HOST(Gra *gra, int ktoryGracz,KodGraya *kod_graya,  float *wynik, float stawka);
extern void grajOgraniczenieStawkiNaWejscieR1HOST(Gra *gra, int ktoryGracz,  KodGraya *kod_graya,  KodGraya *kod_graya2,  float *wynik);
extern void grajWysokieKartyNaWejscieR1HOST(Gra *gra, int ktoryGracz,  KodGraya *kod_graya,  float *wynik, float stawka);
extern void grajBardzoWysokieKartyNaWejscieR1HOST(Gra *gra, int ktoryGracz,  KodGraya *kod_graya,  float *wynik, float stawka);
extern void wymaganychGlosowRXHOST(Gra *gra, int ktoryGracz,  KodGraya *kod_graya,  float *wynik);

extern IleGracR1 *getIleGracR1PTRZReguly(Reguly *regula);
extern StawkaR1 *getStawkaR1PTRZReguly(Reguly *regula);
extern DobijanieR1 *getDobijanieR1PTRZReguly(Reguly *regula);
extern CzyGracR1 *getCzyGracR1PTRZReguly(Reguly *regula);
extern IleGracRX *getIleGracRXPTRZReguly(Reguly *regula, int runda);
extern StawkaRX *getStawkaRXPTRZReguly(Reguly *regula, int runda);
extern DobijanieRX *getDobijanieRXPTRZReguly(Reguly *regula, int runda);
extern CzyGracRX *getCzyGracRXPTRZReguly(Reguly *regula, int runda);
extern Reguly *getReguly();


extern void aplikujIleGracR1HOST(IleGracR1 *reguly, Gra *gra, int ktoryGracz, float *output, float stawka);
extern void aplikujIleGracRXHOST(IleGracRX *reguly, Gra *gra, int ktoryGracz, float *output, float stawka);
extern void aplikujStawkaR1HOST(StawkaR1 *reguly, Gra *gra, int ktoryGracz, float *output);
extern void aplikujStawkaRXHOST(StawkaRX *reguly, Gra *gra, int ktoryGracz, float *output);
extern void aplikujDobijanieR1HOST(DobijanieR1 *reguly, Gra *gra, int ktoryGracz, float *output, float stawka);
extern void aplikujDobijanieRXHOST(DobijanieRX *reguly, Gra *gra, int ktoryGracz, float *output, float stawka);
extern void aplikujCzyGracR1HOST(CzyGracR1 *reguly, Gra *gra, int ktoryGracz, float *output, float stawka);
extern void aplikujCzyGracRXHOST(CzyGracRX *reguly, Gra *gra, int ktoryGracz, float *output, float stawka);

extern void stawkaParaWRekuR1HOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya, float *wynik);
extern void stawkaKolorWRekuR1HOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya, float *wynik);
extern void stawkaWysokaKartaWRekuR1HOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya, float *wynik);
extern void stawkaBardzoWysokaKartaWRekuR1HOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya, float *wynik);
extern void stawkaStalaHOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya, float *wynik);
extern void stawkaWysokaKartaRXHOST(Gra *gra, int ktoryGracz, KodGraya *kod_graya, float *wynik, int wymagany_rezultat);
extern void StawkaLicytujGdyMalaHOST(Gra *gra, int ktoryGracz,KodGraya *gray_stawka,KodGraya *limit_stawki, float *wynik);

extern void grajRunde1HOST(float bid, IleGracR1 *ile_grac, StawkaR1 *stawka, DobijanieR1 *dobijanie, CzyGracR1 *czy_grac,
		Gra *gra, int ktory_gracz, float *wynik);
extern void grajRundeXHOST(float bid, IleGracRX *ile_grac, StawkaRX *stawka, DobijanieRX *dobijanie, CzyGracRX *czy_grac, Gra *gra, int ktory_gracz, float *wynik);
extern void getBilans(Gra *gra,  float *output );
%}

%include typemaps.i 

extern void destruktorGra(Gra *IN);
extern void destruktorInt(int *IN);
extern void destruktorKodGraya(KodGraya *IN);
extern void destruktorRozdanie(Rozdanie *IN);
extern void destruktorHand(Hand *IN);

extern void rozegrajNGier(int ktory_nasz, int **IN, float *INOUT, int N, int liczba_intow, int liczba_osobnikow);
extern void rozegrajNGierCUDA(int ktory_nasz, int **IN, float *INOUT, int n, int liczba_intow, int liczba_watkow, int liczba_osobnikow);
extern int **getIndividualPTRPTR(int size);
extern void setIndividualPTR(int *IN, int **IN, int index);

extern int obliczKodGrayaHOST(int *IN, KodGraya *IN);

float rozegrajPartieHOST(Gra *IN, int ktory_gracz, Reguly *reguly);
extern KodGraya *getKodGrayaPTR(int , int );
extern int *getOsobnikPTR(int *INOUT, int);
extern Gra *getGraPtr();

extern void grajHOST(Gra *IN, int indeks_gracza, float *INOUT, Reguly *IN);

extern void setIleGraczyWGrze(Gra *IN, int ile);
extern void setPula(Gra *IN, int ile);
extern void setStawka(Gra *IN, int ile);
extern void setRunda(Gra *IN, int ile);
extern void setMode(Gra *IN, int ile);

extern void ileGracParaWRekuR1HOST(Gra *IN, int, KodGraya *IN, KodGraya *IN, float *INOUT);
extern void ileGracKolorWRekuR1HOST(Gra *IN, int, KodGraya *IN, KodGraya *IN, float *INOUT);
extern void ileGracWysokaKartaWRekuR1HOST(Gra *IN, int, KodGraya *IN, KodGraya *IN, float *INOUT);
extern void ileGracBardzoWysokaKartaWRekuR1HOST(Gra *IN, int, KodGraya *IN, KodGraya *IN, float *INOUT);
extern void IleGracXGraczyWGrzeRXHOST(Gra *IN, int, KodGraya *IN, KodGraya *IN,  float *INOUT, int ileGraczy);
extern void IleGracPulaRXHOST(Gra *IN, int, KodGraya *IN, KodGraya *IN, KodGraya *IN, float *INOUT);
extern void IleGracStawkaRXHOST(Gra *IN, int, KodGraya *IN, KodGraya *IN, KodGraya *IN, float *INOUT);
extern void IleGracRezultatRXHOST(Gra *IN, int, KodGraya *IN, KodGraya *IN, float *INOUT, int);

extern void dobijajZawszeHOST(Gra *IN, int ktoryGracz, int gen_startowy, float *INOUT);
extern void dobijajGdyParaWRekuR1HOST(Gra *IN, int ktoryGracz,  int, float *INOUT, float, float);
extern void dobijajGdyWysokaKartaR1HOST(Gra *IN, int ktoryGracz,  int , float *INOUT, float, float);
extern void dobijajGdyBrakujeXRXHOST(Gra *IN, int ktoryGracz,  int , float *INOUT, float, float);
extern void dobijajGdyWysokaKartaRXHOST(Gra *IN, int ktoryGracz,  int pozycja_genu,  float *INOUT, float stawka, int wymagany_rezultat);

extern void grajRezultatRXHOST(Gra *IN, int ktoryGracz,  KodGraya *IN,  float *INOUT, int wymagany_rezultat);
extern void ograniczenieStawkiRXHOST(Gra *IN, int ktoryGracz,  KodGraya *IN, KodGraya *IN,  float *INOUT);
extern void grajGdyParaWRekuR1HOST(Gra *IN, int ktoryGracz,KodGraya *IN,  float *INOUT, float stawka);
extern void grajGdyKolorWRekuR1HOST(Gra *IN, int ktoryGracz,KodGraya *IN,  float *INOUT, float stawka);
extern void grajOgraniczenieStawkiNaWejscieR1HOST(Gra *IN, int ktoryGracz,  KodGraya *IN,  KodGraya *IN,  float *INOUT);
extern void grajWysokieKartyNaWejscieR1HOST(Gra *gra, int ktoryGracz, KodGraya *IN,  float *INOUT, float stawka);
extern void grajBardzoWysokieKartyNaWejscieR1HOST(Gra *gra, int ktoryGracz, KodGraya *IN,  float *INOUT, float stawka);
extern void wymaganychGlosowRXHOST(Gra *gra, int ktoryGracz,  KodGraya *IN,  float *INOUT);

extern IleGracR1 *getIleGracR1PTRZReguly(Reguly *IN);
extern StawkaR1 *getStawkaR1PTRZReguly(Reguly *IN);
extern DobijanieR1 *getDobijanieR1PTRZReguly(Reguly *IN);
extern CzyGracR1 *getCzyGracR1PTRZReguly(Reguly *IN);
extern IleGracRX *getIleGracRXPTRZReguly(Reguly *IN, int runda);
extern StawkaRX *getStawkaRXPTRZReguly(Reguly *IN, int runda);
extern DobijanieRX *getDobijanieRXPTRZReguly(Reguly *IN, int runda);
extern CzyGracRX *getCzyGracRXPTRZReguly(Reguly *IN, int runda);
extern Reguly *getReguly();

extern void aplikujIleGracR1HOST(IleGracR1 *IN, Gra *IN, int, float *INOUT, float);
extern void aplikujIleGracRXHOST(IleGracRX *IN, Gra *IN, int, float *INOUT, float);
extern void aplikujStawkaR1HOST(StawkaR1 *IN, Gra *IN, int, float *INOUT);
extern void aplikujStawkaRXHOST(StawkaRX *IN, Gra *IN, int, float *INOUT);
extern void aplikujDobijanieR1HOST(DobijanieR1 *IN, Gra *IN, int, float *INOUT, float);
extern void aplikujDobijanieRXHOST(DobijanieRX *IN, Gra *IN, int, float *INOUT, float);
extern void aplikujCzyGracR1HOST(CzyGracR1 *IN, Gra *IN, int, float *INOUT, float);
extern void aplikujCzyGracRXHOST(CzyGracRX *IN, Gra *IN, int, float *INOUT, float);

extern void stawkaParaWRekuR1HOST(Gra *IN, int, KodGraya *IN, float *INOUT);
extern void stawkaKolorWRekuR1HOST(Gra *IN, int, KodGraya *IN, float *INOUT);
extern void stawkaWysokaKartaWRekuR1HOST(Gra *IN, int, KodGraya *IN, float *INOUT);
extern void stawkaBardzoWysokaKartaWRekuR1HOST(Gra *IN, int, KodGraya *IN, float *INOUT);
extern void stawkaStalaHOST(Gra *IN, int, KodGraya *IN, float *INOUT);
extern void stawkaWysokaKartaRXHOST(Gra *IN, int ktoryGracz, KodGraya *IN, float *INOUT, int wymagany_rezultat);
extern void StawkaLicytujGdyMalaHOST(Gra *IN, int ktoryGracz,KodGraya *IN,KodGraya *IN,  float *INOUT);

extern void grajRunde1HOST(float bid, IleGracR1 *IN, StawkaR1 *IN, DobijanieR1 *IN, CzyGracR1 *IN,
			Gra *IN, int ktory_gracz, float *INOUT);
extern void grajRundeXHOST(float bid, IleGracRX *IN, StawkaRX *IN, DobijanieRX *IN, CzyGracRX *IN,
			Gra *IN, int ktory_gracz, float *INOUT);
extern void getBilans(Gra *IN,  float *INOUT );