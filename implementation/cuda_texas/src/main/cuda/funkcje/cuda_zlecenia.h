#include "../struktury/gra.h"
#include "../struktury/zlecenie.h"


#ifndef CUDA_ZLECENIA_H
#define CUDA_ZLECENIA_H

extern "C" {

	Zlecenie *noweZlecenie(int osobnik1, int osobnik2, int osobnik3, int osobnik4, int osobnik5,  int osobnik6,
								int ktory_nasz, int nr_rozdania, int *osobniki) {

			Zlecenie *zlecenie = new Zlecenie();
			zlecenie->osobniki = osobniki;
			zlecenie->indexOsobnika[0] = osobnik1;
			zlecenie->indexOsobnika[1] = osobnik2;
			zlecenie->indexOsobnika[2] = osobnik3;
			zlecenie->indexOsobnika[3] = osobnik4;
			zlecenie->indexOsobnika[4] = osobnik5;
			zlecenie->indexOsobnika[5] = osobnik6;
			zlecenie->indexGracza=ktory_nasz;
			zlecenie->nrRozdania=nr_rozdania;

			return zlecenie;
	};

}

#endif
