package reguly.dobijanie;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.RegulaAbstrakcyjnaDobijania;
import engine.Gra;
import engine.rezultaty.Rezultat;

/**
 * Regula zwraca true, jesli jest wlaczona
 * 
 * @author Kacper Gorski
 *
 */
public class RegulaDobijacZawsze extends RegulaAbstrakcyjnaDobijania {

  public RegulaDobijacZawsze(int pozycjaStartowaWGenotypie) {
    super(pozycjaStartowaWGenotypie, 1);
  }

  @Override
  public double aplikujRegule(Gra gra, int kolejnosc, EvBinaryVectorIndividual osobnik, Rezultat rezultat) {
    if (osobnik.getGene(pozycjaStartowaWGenotypie) == 1 )
      return 1.0d;
    else 
      return 0.0d;
  }

  @Override
  public void zmienIndividuala(double[] argumenty,
      EvBinaryVectorIndividual individual) {
    
    if (argumenty[0] == 1.0d)
      individual.setGene(pozycjaStartowaWGenotypie, 1);
    else
      individual.setGene(pozycjaStartowaWGenotypie, 0);
  }

}
