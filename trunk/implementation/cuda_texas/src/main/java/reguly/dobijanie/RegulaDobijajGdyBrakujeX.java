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
public class RegulaDobijajGdyBrakujeX extends RegulaAbstrakcyjnaDobijania {

  // oznacza jaki procent stawki proponowanej przez gracza do gry jest wymagany, aby dobic
  public double wspolczynnikDobijania;
  
  public RegulaDobijajGdyBrakujeX(int pozycjaStartowaWGenotypie, double aWspolczynnikDobijania) {
    super(pozycjaStartowaWGenotypie, 1);
    wspolczynnikDobijania = aWspolczynnikDobijania;
  }
 
  public double aplikujRegule(Gra gra, int aKolejnosc, EvBinaryVectorIndividual osobnik, Rezultat rezultat) {
    if (osobnik.getGene(pozycjaStartowaWGenotypie) == 1 &&
        stawka <= gra.stawka * wspolczynnikDobijania)
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
    
    wspolczynnikDobijania = argumenty[1];
    
  }

}

