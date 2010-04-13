package Gracze;

import engine.Gra;

public abstract class Gracz {

	public double bilans = 0;

	public double musik = 0;

	public Gra gra = null;

	// -1 pass lub ilosc pieniedzy
	public abstract double play(int i, double j);

	public void wygrana(double ilosc) {

		bilans += ilosc;
	}

	public double bilans() {
		return bilans;
	}

	public void musik(double wielkosc_musika) {
		musik = wielkosc_musika;
		bilans -= musik;
	}

	public String toString() {
		return "bilans:" + bilans;
	}

}
