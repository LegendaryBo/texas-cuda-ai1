%module texas_swig
%{
#include "../../../src/main/java/c/struktury/texas_struktury.h"
#include "../../../src/main/java/c/struktury/gra.h"
#include "../../../src/main/java/c/struktury/gracz.h"

extern void iloscKartTejSamejWysokosciHOST(Hand *, int *);
extern Hand *alokujObiekt(int * , int ileKart);
extern int jestStreetHOST(Hand *reka);
extern int jestKolorHOST(Hand *reka);
extern int jestPokerHOST(Hand *reka);
extern int najlepszaKartaHOST(Hand *reka);
extern void najlepszaKartaRozdaniaHOST(Rozdanie *reka, int *INOUT);
extern void generuj(int rozdanie, Rozdanie *rozd, int *INOUT);
extern Rozdanie *gerRozdaniePtr();
extern void sprawdzRezultatyHOST(Rozdanie *rozdanie);
extern int porownaj(Rozdanie *rozdanie, int gracz1, int gracz2);

extern int wygrany(Rozdanie *rozdanie, int *rozpatrywani, int *wygrani);

extern Gra *getGraPTR();
extern void nowaGra(int *gracz1_geny, int *gracz2_geny, int *gracz3_geny,
		int *gracz4_geny, int *gracz5_geny, int *gracz6_geny,
		int nr_rozdania, int mode, Gra *gra);
extern float rozegrajPartieHOST(Gra *gra, int indeksGracza);

%}
 
%include typemaps.i
extern void iloscKartTejSamejWysokosciHOST(Hand *IN, int *INOUT);
extern Hand *alokujObiekt(int *INOUT, int);
extern int jestStreetHOST(Hand *IN);
extern int jestKolorHOST(Hand *IN);
extern int jestPokerHOST(Hand *IN);
extern int najlepszaKartaHOST(Hand *IN);
extern void najlepszaKartaRozdaniaHOST(Rozdanie *IN, int *INOUT);
extern void generuj(int rozdanie, Rozdanie *rozd, int *INOUT);
extern Rozdanie *gerRozdaniePtr();
extern void sprawdzRezultatyHOST(Rozdanie *IN);
extern int porownaj(Rozdanie *IN, int gracz1, int gracz2);

extern int wygrany(Rozdanie *IN, int *INOUT, int *INOUT);

extern Gra *getGraPTR();
extern void nowaGra(int *IN, int *IN, int *IN,
		int *IN, int *IN, int *IN, int, int, Gra *gra);
extern float rozegrajPartieHOST(Gra *IN, int indeksGracza);

