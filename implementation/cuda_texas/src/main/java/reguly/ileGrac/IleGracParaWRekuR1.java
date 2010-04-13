package reguly.ileGrac;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.RegulaAbstrakcyjnaIleGrac;
import reguly.kodGraya.KodGraya;
import engine.Gra;
import engine.rezultaty.Rezultat;

/**
 * Regula TYLKO dla rundy 1
 * 
 * Odpowiada za strategie przy stawianiu stawek.
 * 
 */
public class IleGracParaWRekuR1 extends RegulaAbstrakcyjnaIleGrac {

	// FI - wspolczynnik 0-1, ktory mnozy sie przez maksymalna mozliwa
	// stawke

	public KodGraya gray_fi;
	private final int DLUGOSC_FI = 5;
	private final int WIELKOSC_FI = 31;

	public IleGracParaWRekuR1(int pozycjaStartowaWGenotypie,
			int dlugosc_wagi) {
		super(pozycjaStartowaWGenotypie, 1 + dlugosc_wagi + 5,
				dlugosc_wagi);

		gray_fi = new KodGraya(DLUGOSC_FI, pozycjaStartowaWGenotypie
				+ dlugosc_wagi + 1);

	}

	@Override
	public double aplikujRegule(Gra gra, int kolejnosc,
			EvBinaryVectorIndividual osobnik, Rezultat rezultat) {
//		System.out.println("wartosc kodu graya_fi java "+gray_fi.getWartoscKoduGraya(osobnik));
//		System.out.println("startowa kodu graya_fi java "+gray_fi.getPozycjaStartowa());
//		System.out.println("wartosc kodu graya_wartosc java "+kodGrayaWagi.getWartoscKoduGraya(osobnik));
//		System.out.println("startowa kodu graya_wartosc java "+kodGrayaWagi.getPozycjaStartowa());
//		System.out.print("Osobnik: ");
//		for (int i=0; i < gray_fi.getDlugoscKodu(); i++)
//			System.out.print( osobnik.getGene(gray_fi.getPozycjaStartowa() + i) );
//		System.out.print("\n");
		
		
		
		if (gra.getPrivateCard(kolejnosc, 0).wysokosc == gra
				.getPrivateCard(kolejnosc, 1).wysokosc) {
			if (osobnik.getGene(pozycjaStartowaWGenotypie) == 1) {
//				System.out.println("rowne i wlaczone c");
				
				return (2.0d / WIELKOSC_FI) * gray_fi.getWartoscKoduGraya(osobnik);
			}
			else {
//				System.out.println("rowne i wylaczone c");
				return -1.0d;
			}
		} else {
//			System.out.println("nierowne c");
			return -1.0d;
		}
	}

}
