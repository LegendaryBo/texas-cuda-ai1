package generator;

import engine.Hand;
import engine.Karta;

/**
 * 
 * Obiekt odpowiedzialny za generowanie rozdan do texas holdem.
 * 
 * Mozna ustawic seed
 * 
 * @author Kacper Gorski (railman85@gmail.com)
 * 
 */
public class GeneratorRozdan {

	//private Random generator = null;
	int prev_seed;

	private CustomGenerator generator = new CustomGenerator();
	
	// karty w rozdaniu
	// 5 kart publicznych
	public Karta[] publiczny_stos = new Karta[5];

	// po 2 karty na reke dla kazdego
	public Karta[] gracz1 = new Karta[2];
	public Karta[] gracz2 = new Karta[2];
	public Karta[] gracz3 = new Karta[2];
	public Karta[] gracz4 = new Karta[2];
	public Karta[] gracz5 = new Karta[2];
	public Karta[] gracz6 = new Karta[2];

	/**
	 * Tworzy obiekt z lopsowym seeden
	 */
	public GeneratorRozdan() {
		//generator = new Random((new Date()).getTime());
		generate();
	}

	/**
	 * Tworzy obiekt z podanym seedem
	 * 
	 * @param rozdanie
	 *                - seed rozdania
	 */
	public GeneratorRozdan(int rozdanie) {
		//generator = new Random(rozdanie);
		generator = new CustomGenerator(rozdanie);
		
		generate();

	}

	// metoda zwraca wszystkie 2 karty prywatne + 5 publicznych dla podanego
	// indexu gracza
	public Karta[] getAllCards(int numer_gracza) {
		Karta[] ret = new Karta[7];
		for (int i = 0; i < 5; i++)
			ret[i] = publiczny_stos[i];

		if (numer_gracza == 0) {
			ret[5] = gracz1[0];
			ret[6] = gracz1[1];
		}
		if (numer_gracza == 1) {
			ret[5] = gracz2[0];
			ret[6] = gracz2[1];
		}
		if (numer_gracza == 2) {
			ret[5] = gracz3[0];
			ret[6] = gracz3[1];
		}
		if (numer_gracza == 3) {
			ret[5] = gracz4[0];
			ret[6] = gracz4[1];
		}
		if (numer_gracza == 4) {
			ret[5] = gracz5[0];
			ret[6] = gracz5[1];
		}
		if (numer_gracza == 5) {
			ret[5] = gracz6[0];
			ret[6] = gracz6[1];
		}
		return ret;
	}

	/**
	 * Output rozdania
	 */
	public String toString() {

		String ret = new String();
		ret = "publiczne karty:";

		for (int i = 0; i < 5; i++) {
			ret += " & " + publiczny_stos[i];
		}

		ret += "\n";
		ret += "gracz1:";
		for (int i = 0; i < 2; i++) {
			ret += " & " + gracz1[i];
		}

		ret += "\n";
		ret += "gracz2:";
		for (int i = 0; i < 2; i++) {
			ret += " & " + gracz2[i];
		}

		ret += "\n";
		ret += "gracz3:";
		for (int i = 0; i < 2; i++) {
			ret += " & " + gracz3[i];
		}

		ret += "\n";
		ret += "gracz4:";
		for (int i = 0; i < 2; i++) {
			ret += " & " + gracz4[i];
		}

		ret += "\n";
		ret += "gracz5:";
		for (int i = 0; i < 2; i++) {
			ret += " & " + gracz5[i];
		}

		ret += "\n";
		ret += "gracz6:";
		for (int i = 0; i < 2; i++) {
			ret += " & " + gracz6[i];
		}
		return ret;
	}

	// ************* generowanie rozdania ***************
	public void generate() {

		prev_seed = generator.getSeed();
		
		// w tym algorytmie dodajemy do arraylisty wszystkie karty i
		// wybieramy 11 z nich

		//ArrayList<Karta> stos = new ArrayList<Karta>();
		Karta[] talia = new Karta[52];
		
		for (int i = 2; i <= 14; i++) {
			for (int j = 1; j <= 4; j++) {
				talia[(j-1)*13 + i -2 ] = new Karta(i,j);
			}
		}

		int kart = 4 * 13;

		for (int i = 0; i < 5; i++) {
			int numer_karty = generator.nextInt(kart);
			//System.out.println(numer_karty);
			publiczny_stos[i] = talia[numer_karty];
			kart--;
			talia[numer_karty] = talia[kart];
		}

		for (int i = 0; i < 2; i++) {
			int numer_karty = generator.nextInt(kart);
			gracz1[i] = talia[numer_karty];
			kart--;
			talia[numer_karty] = talia[kart];
		}

		for (int i = 0; i < 2; i++) {
			int numer_karty = generator.nextInt(kart);
			gracz2[i] = talia[numer_karty];
			kart--;
			talia[numer_karty] = talia[kart];
		}

		for (int i = 0; i < 2; i++) {
			int numer_karty = generator.nextInt(kart);
			gracz3[i] = talia[numer_karty];
			kart--;
			talia[numer_karty] = talia[kart];
		}

		for (int i = 0; i < 2; i++) {
			int numer_karty = generator.nextInt(kart);
			gracz4[i] = talia[numer_karty];
			kart--;
			talia[numer_karty] = talia[kart];
		}

		for (int i = 0; i < 2; i++) {
			int numer_karty = generator.nextInt(kart);
			gracz5[i] = talia[numer_karty];
			kart--;
			talia[numer_karty] = talia[kart];
		}

		for (int i = 0; i < 2; i++) {
			int numer_karty = generator.nextInt(kart);
			gracz6[i] = talia[numer_karty];
			kart--;
			talia[numer_karty] = talia[kart];
		}
	}


	public Hand getHand(int ktoryGracz) {
		return new Hand(getAllCards(ktoryGracz));
	}

	public int getSeed() {
		
		return prev_seed;
	}


}
