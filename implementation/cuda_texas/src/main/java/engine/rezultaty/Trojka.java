package engine.rezultaty;

import engine.Karta;

public class Trojka extends Rezultat {

	public int poziom_trojki = 0;
	public int karta1 = 0;
	public int karta2 = 0;

	@Override
	public int porownaj_take_same(Rezultat rezultat) {
		Trojka przeciwnik = (Trojka) rezultat;

		if (poziom_trojki > przeciwnik.poziom_trojki)
			return 1;
		if (poziom_trojki < przeciwnik.poziom_trojki)
			return -1;
		if (karta1 > przeciwnik.karta1)
			return 1;
		if (karta1 < przeciwnik.karta1)
			return -1;
		if (karta2 > przeciwnik.karta2)
			return 1;
		if (karta2 < przeciwnik.karta2)
			return -1;

		return 0;
	}

	public Trojka(Karta[] karta) {

		poziom = 4;

		int[] count = new int[15];
		for (int i = 0; i < 7; i++) {
			count[karta[i].wysokosc]++;
		}

		int obliczone = 0;

		for (int i = 14; i >= 2; i--) {

			if (count[i] == 3)
				poziom_trojki = i;

			if (count[i] == 1 && obliczone != 2) {
				if (obliczone == 0)
					karta1 = i;
				if (obliczone == 1)
					karta2 = i;

				obliczone++;
			}

		}
	}

	public String toString() {
		return "Trojka " + poziom_trojki + ". Reszta: " + karta1 + ","
				+ karta2;
	}

}
