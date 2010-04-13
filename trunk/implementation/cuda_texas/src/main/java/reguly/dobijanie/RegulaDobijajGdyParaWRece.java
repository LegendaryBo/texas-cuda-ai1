package reguly.dobijanie;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.RegulaAbstrakcyjnaDobijania;
import engine.Gra;
import engine.rezultaty.Rezultat;

public class RegulaDobijajGdyParaWRece extends RegulaAbstrakcyjnaDobijania {

	private double wspolczynnikDobijania = 0.0d;

	public RegulaDobijajGdyParaWRece(int pozycjaStartowaWGenotypie,
			double aWspolczynnikDobijania) {
		super(pozycjaStartowaWGenotypie, 1);
		wspolczynnikDobijania = aWspolczynnikDobijania;
	}

	@Override
	public double aplikujRegule(Gra gra, int kolejnosc,
			EvBinaryVectorIndividual osobnik, Rezultat rezultat) {

		if (gra.getPrivateCard(kolejnosc, 0).wysokosc == gra
				.getPrivateCard(kolejnosc, 1).wysokosc) {
			if (osobnik.getGene(pozycjaStartowaWGenotypie) == 1
					&& stawka < gra.stawka
							* wspolczynnikDobijania)
				return 1.0d;
			else {
//				System.out.println("niespelnione");
				return 0.0d;
			}
		} else {
			return 0.0d;
		}
	}

	@Override
	public void zmienIndividuala(double[] argumenty,
			EvBinaryVectorIndividual individual) {

		if (argumenty[0] == 1.0d)
			individual.setGene(pozycjaStartowaWGenotypie, 1);
		else
			individual.setGene(pozycjaStartowaWGenotypie, 0);

		wspolczynnikDobijania = (int) argumenty[1];

	}

}
