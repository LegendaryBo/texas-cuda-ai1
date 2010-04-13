package reguly.ileGrac;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.RegulaAbstrakcyjnaIleGrac;
import reguly.kodGraya.KodGraya;
import engine.Gra;
import engine.rezultaty.Rezultat;

public class IleGracWysokaKartaWRekuR1 extends RegulaAbstrakcyjnaIleGrac {

	// FI - wspolczynnik 0-1, ktory mnozy sie przez maksymalna mozliwa
	// stawke

	public KodGraya gray_fi;
	private final int DLUGOSC_FI = 5;
	private final int WIELKOSC_FI = 31;

	public IleGracWysokaKartaWRekuR1(int pozycjaStartowaWGenotypie,
			int dlugosc_wagi) {
		super(pozycjaStartowaWGenotypie, 1 + dlugosc_wagi + 5,
				dlugosc_wagi);

		gray_fi = new KodGraya(DLUGOSC_FI, pozycjaStartowaWGenotypie
				+ dlugosc_wagi + 1);

	}

	@Override
	public double aplikujRegule(Gra gra, int kolejnosc,
			EvBinaryVectorIndividual osobnik, Rezultat rezultat) {

//		System.out.println("java, wysokosc 1 karty "+gra.getPrivateCard(kolejnosc, 0).wysokosc);
//		System.out.println("java, wysokosc 2 karty "+gra.getPrivateCard(kolejnosc, 1).wysokosc);
		
		if (gra.getPrivateCard(kolejnosc, 0).wysokosc >= 10
				&& gra.getPrivateCard(kolejnosc, 1).wysokosc >= 10) {
			if (osobnik.getGene(pozycjaStartowaWGenotypie) == 1)
				return (2.0d / WIELKOSC_FI)
						* gray_fi.getWartoscKoduGraya(osobnik);
			else {
				return -1.0d;
			}
		} else {
			return -1.0d;
		}

	}

}
