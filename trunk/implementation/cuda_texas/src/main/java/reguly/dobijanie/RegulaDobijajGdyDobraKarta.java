package reguly.dobijanie;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.RegulaAbstrakcyjnaDobijania;
import reguly.kodGraya.KodGraya;
import engine.Gra;
import engine.rezultaty.Rezultat;

public class RegulaDobijajGdyDobraKarta extends RegulaAbstrakcyjnaDobijania {

  private int rezultat = 1;
  
  public RegulaDobijajGdyDobraKarta(int pozycjaStartowaWGenotypie, int aRezultat) {
    super(pozycjaStartowaWGenotypie, 1);
   rezultat = aRezultat;
  }

  @Override
  public double aplikujRegule(Gra aGra, int aKolejnosc, EvBinaryVectorIndividual osobnik, Rezultat aRezultat) {
    // jesli jest para
    
    if (aRezultat.poziom >= rezultat) {
  
      // jesli regula jest wlaczona
      if (osobnik.getGene(pozycjaStartowaWGenotypie) == 1 )
        return 1.0d; 
      else 
        return 0.0d;
      
    }
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
    
    rezultat = (int) argumenty[1];

    
  }

}
