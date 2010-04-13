package engine.rezultaty;

import engine.Karta;

public class Smiec extends Rezultat {

	// karty[0] - najwyzsza
	public byte[] karty = new byte[5];

	@Override
	public int porownaj_take_same(Rezultat rezultat) {
		Smiec przeciwnik = (Smiec) rezultat;

		if (karty[0] > przeciwnik.karty[0])
			return 1;
		if (karty[0] < przeciwnik.karty[0])
			return -1;
		if (karty[1] > przeciwnik.karty[1])
			return 1;
		if (karty[1] < przeciwnik.karty[1])
			return -1;
		if (karty[2] > przeciwnik.karty[2])
			return 1;
		if (karty[2] < przeciwnik.karty[2])
			return -1;
		if (karty[3] > przeciwnik.karty[3])
			return 1;
		if (karty[3] < przeciwnik.karty[3])
			return -1;
		if (karty[4] > przeciwnik.karty[4])
			return 1;
		if (karty[4] < przeciwnik.karty[4])
			return -1;

		return 0;

	}

	public Smiec(Karta[] karta) {

		poziom = 1;

		byte[] count = new byte[15];
		for (byte i = 0; i < 7; i++) {
			count[karta[i].wysokosc]++;
		}

		byte obliczone = 0;

		for (byte i = 14; i >= 2; i--) {

			if (count[i] > 0) {
				karty[obliczone] = i;
				obliczone++;
			}
			if (obliczone == 5)
				return;

		}

	}

	public String toString() {
		String ret = "Smiec:";
		for (int i = 0; i < 5; i++) {
			ret += " " + karty[i];
		}

		return ret;
	}

}
