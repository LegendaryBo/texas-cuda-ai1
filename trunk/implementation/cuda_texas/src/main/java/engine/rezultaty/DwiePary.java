package engine.rezultaty;

import engine.Karta;

public class DwiePary extends Rezultat {

	public byte nizsza_para = 0;
	public byte wyzsza_para = 0;
	public byte najwyzsza_karta = 0;

	@Override
	public int porownaj_take_same(Rezultat rezultat) {
		DwiePary przeciwnik = (DwiePary) rezultat;

		if (wyzsza_para > przeciwnik.wyzsza_para)
			return 1;
		if (wyzsza_para < przeciwnik.wyzsza_para)
			return -1;
		if (nizsza_para > przeciwnik.nizsza_para)
			return 1;
		if (nizsza_para < przeciwnik.nizsza_para)
			return -1;
		if (najwyzsza_karta > przeciwnik.najwyzsza_karta)
			return 1;
		if (najwyzsza_karta < przeciwnik.najwyzsza_karta)
			return -1;

		return 0;
	}

	public DwiePary(Karta[] karta) {

		poziom = 3;

		byte[] count = new byte[15];
		for (byte i = 0; i < 7; i++) {
			count[karta[i].wysokosc]++;
		}

		byte obliczone = 0;

		for (byte i = 14; i >= 2; i--) {

			if (count[i] == 1 && i > najwyzsza_karta)
				najwyzsza_karta = i;
			if (count[i] == 2 && obliczone == 1) {
				nizsza_para = i;
				obliczone++;
			}
			if (count[i] == 2 && obliczone == 0) {
				wyzsza_para = i;
				obliczone++;
			}

		}
	}

	public String toString() {

		return "Dwie pary: " + wyzsza_para + " na " + nizsza_para
				+ " oraz " + najwyzsza_karta;

	}

}
