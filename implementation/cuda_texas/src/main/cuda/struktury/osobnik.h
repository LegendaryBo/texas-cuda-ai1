
#ifndef OSOBNIK_STRUKTURY_H
#define OSOBNIK_STRUKTURY_H

typedef struct {
	int *geny;
	int dlugoscGenow;
} Osobnik;

typedef struct {
	Osobnik **osobniki;
	int liczba_osobnikow;
} ZbiorOsobnikow;

#endif
