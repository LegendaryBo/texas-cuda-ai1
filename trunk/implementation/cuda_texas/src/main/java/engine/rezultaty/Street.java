package engine.rezultaty;

import engine.Karta;

public class Street extends Rezultat {

	public int najwyzsza_karta = 0;

	@Override
	public int porownaj_take_same(Rezultat rezultat) {
		Street przeciwnik = (Street) rezultat;

		if (najwyzsza_karta > przeciwnik.najwyzsza_karta)
			return 1;
		if (najwyzsza_karta < przeciwnik.najwyzsza_karta)
			return -1;

		return 0;
	}

	public Street(Karta[] karty) {

		poziom = 5;

		int[] count = new int[15];
		for (int i = 0; i < 7; i++) {
			count[karty[i].wysokosc]++;
		}
		count[1]+=count[14];

		int dno_streeta = 1;
		int szczyt_streeta = 1;

		for (int i = 1; i < 15; i++) {

			if (count[i] > 0)
				szczyt_streeta = i;
			else
				dno_streeta = i + 1;

			if (szczyt_streeta - dno_streeta >= 4) {
				najwyzsza_karta = i;
			}
		}

	}

}
