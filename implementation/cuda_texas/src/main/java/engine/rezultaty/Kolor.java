package engine.rezultaty;

import engine.Karta;

public class Kolor extends Rezultat {

	public int najwyzsza_karta = 0;

	@Override
	public int porownaj_take_same(Rezultat rezultat) {
		Kolor przeciwnik = (Kolor) rezultat;

		if (najwyzsza_karta > przeciwnik.najwyzsza_karta)
			return 1;
		if (najwyzsza_karta < przeciwnik.najwyzsza_karta)
			return -1;

		return 0;
	}

	public Kolor(Karta[] karty) {

		poziom = 6;

		int[] count = new int[5];
		for (int i = 0; i < 7; i++) {
			count[karty[i].kolor]++;
		}

		int max_kolor = 0;
		for (int i = 1; i <= 4; i++) {
			if (count[i] >= 5)
				max_kolor = i;
		}

		for (int i = 0; i < 7; i++) {
			if (karty[i].wysokosc > najwyzsza_karta
					&& karty[i].kolor == max_kolor)
				najwyzsza_karta = karty[i].wysokosc;
		}

	}

}
