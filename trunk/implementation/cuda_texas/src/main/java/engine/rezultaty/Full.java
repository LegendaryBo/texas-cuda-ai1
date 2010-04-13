package engine.rezultaty;

import engine.Karta;

public class Full extends Rezultat {

	public int trojka = 0;
	public int dwojka = 0;

	@Override
	public int porownaj_take_same(Rezultat rezultat) {
		Full przeciwnik = (Full) rezultat;

		if (trojka > przeciwnik.trojka)
			return 1;
		if (trojka < przeciwnik.trojka)
			return -1;
		if (dwojka > przeciwnik.dwojka)
			return 1;
		if (dwojka < przeciwnik.dwojka)
			return -1;

		return 0;
	}

	public Full(Karta[] karta) {

		poziom = 7;

		int[] count = new int[15];
		for (int i = 0; i < 7; i++) {
			count[karta[i].wysokosc]++;
		}

		for (int i = 14; i >= 2; i--) {

			if (count[i] == 3)
				trojka = i;

			if (count[i] == 2 && i > dwojka) {
				dwojka = i;

			}

		}
	}

}
