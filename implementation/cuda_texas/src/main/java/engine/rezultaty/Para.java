package engine.rezultaty;

import engine.Karta;

public class Para extends Rezultat {

	public byte poziom_pary = 0;
	public byte[] karty = new byte[3];

	@Override
	public int porownaj_take_same(Rezultat rezultat) {
		Para przeciwnik = (Para) rezultat;
		if (poziom_pary > przeciwnik.poziom_pary)
			return 1;
		if (poziom_pary < przeciwnik.poziom_pary)
			return -1;
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

		return 0;

	}

	public Para(Karta[] karta) {

		poziom = 2;

		byte[] count = new byte[15];
		for (byte i = 0; i < 7; i++) {
			count[karta[i].wysokosc]++;
		}

		byte obliczone = 0;

		for (byte i = 14; i >= 2; i--) {

			if (count[i] == 2)
				poziom_pary = i;
			if (count[i] == 1 && obliczone != 3) {
				karty[obliczone] = i;
				obliczone++;
			}

		}
	}

	public String toString() {
		String ret = "Para " + poziom_pary + ". Reszta:";
		for (int i = 0; i < 3; i++) {
			ret += " " + karty[i];
		}

		return ret;
	}

}
