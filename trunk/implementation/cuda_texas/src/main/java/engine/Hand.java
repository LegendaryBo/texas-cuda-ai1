package engine;

import cuda.swig.SWIGTYPE_p_Hand;
import cuda.swig.texas_swig;

/**
 * Klasa reprezentuje zestaw kart, ktore ma gracz w reku, wraz z jego kartami
 * prywatnymi (max 7)
 * 
 * @author railman
 * 
 */
public class Hand {

	private int iloscKart = 0;

	private Karta[] karty = new Karta[7];

	public Hand(Karta[] karty) {
		if (karty.length > 7)
			throw new IllegalArgumentException(
					"Podano zbyt duza ilosc kart: " + karty.length);

		for (int i = 0; i < karty.length; i++) {
			this.karty[i] = karty[i];
		}
		iloscKart = karty.length;
	}

	/**
	 * Konweruje karty do postaci tablicy intow.
	 * 
	 * @return tablice, w ktorej: - elementy tablicy 0-6 oznaczaja wysokosci
	 *         kolejnych kart - elementy tablicy 7-14 oznaczaja kolory
	 *         kolejnych kart
	 */
	private int[] toIntArray() {

		int[] tablicaInt = new int[14];
		for (int i = 0; i < iloscKart; i++) {
			tablicaInt[i] = karty[i].wysokosc;
			tablicaInt[i + 7] = karty[i].kolor;
		}

		return tablicaInt;
	}

	/**
	 * Tworzy podany obiekt w pamieci programu C++ i zwraca do niego wskaznik
	 * 
	 * @return
	 */
	public SWIGTYPE_p_Hand stworzObiektWSwigu() {
		return texas_swig.alokujObiekt(toIntArray(), iloscKart);
	}

	/**
	 * Zwraca tablice kart.
	 * 
	 * @return
	 */
	public Karta[] getKarty() {
		return karty;
	}
	
	public String toString() {
		String bla = "";
		
		for (int i=0; i <iloscKart; i++)
			bla += "<"+karty[i]+">";
		
		return bla;
	}

}
