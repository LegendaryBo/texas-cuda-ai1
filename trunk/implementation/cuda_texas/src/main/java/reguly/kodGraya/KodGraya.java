package reguly.kodGraya;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;

/**
 * 
 * To jest metoda odpowiedzialna za tzw. kod graya.
 * 
 * Jest to kodowanie liczby binarnej, w ktorej zmienienie dowolnego bitu
 * powoduje niewielka zmiane wartosc liczby
 * 
 * @author Kacper Gorski
 * 
 */
public class KodGraya {

	private int pozycjaStartowaGenu;

	public int getPozycjaStartowa() {
		return pozycjaStartowaGenu;
	}

	public int getDlugoscKodu() {
		return dlugoscKodu;
	}

	private int dlugoscKodu;

	public KodGraya(int aDlugoscKodu, int aPozycjaStartowaGenu) {
		dlugoscKodu = aDlugoscKodu;
		pozycjaStartowaGenu = aPozycjaStartowaGenu;
	}

	/**
	 * 
	 * @return wartosc zakodowana w kodzie graya
	 * 
	 * 
	 *         Zaimplementowany jest ten:
	 *         http://pl.wikipedia.org/wiki/Kod_Graya algorytm
	 * 
	 */
	final public int getWartoscKoduGraya(
			final EvBinaryVectorIndividual individual) {
		int pLiczbaBinarna = 0;
		int pBit = individual.getGene(pozycjaStartowaGenu);
		
		pLiczbaBinarna += pBit << dlugoscKodu - 1;

		for (int i = 1; i < dlugoscKodu; i++) {
			pBit = (individual.getGene(pozycjaStartowaGenu + i) ^ pBit);
	
			pLiczbaBinarna += pBit << (dlugoscKodu - i - 1);

		}

		return pLiczbaBinarna;
	}

	public void setValue(EvBinaryVectorIndividual individual, int wartosc) {
		int kodGraya = wartosc ^ (wartosc / 2);

		for (int i = 0; i < dlugoscKodu; i++) {

			if (kodGraya % (2 << i) >= 1 << i) {
				individual.setGene(pozycjaStartowaGenu
						+ dlugoscKodu - i - 1, 1);
			} else {
				individual.setGene(pozycjaStartowaGenu
						+ dlugoscKodu - i - 1, 0);
			}
		}
		
	}

}
