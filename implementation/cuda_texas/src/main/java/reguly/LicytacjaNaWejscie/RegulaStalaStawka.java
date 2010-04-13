package reguly.LicytacjaNaWejscie;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.RegulaAbstrakcyjna;
import reguly.kodGraya.KodGraya;
import engine.Gra;
import engine.rezultaty.Rezultat;

/**
 * 
 * Zwraca zwraca stala stawke
 * 
 * @author Kacper Gorski
 *
 */
public class RegulaStalaStawka extends RegulaAbstrakcyjna {

  private KodGraya kodGraya = null; 
  
  public RegulaStalaStawka(int pozycjaStartowaWGenotypie,
      EvBinaryVectorIndividual osobnik, int aDlugoscKoduGraya) {
    super(pozycjaStartowaWGenotypie, 1 + aDlugoscKoduGraya);
    
    kodGraya = new KodGraya(aDlugoscKoduGraya ,pozycjaStartowaWGenotypie + 1);
    
  }

  @Override
  public double aplikujRegule(Gra gra, int kolejnosc, EvBinaryVectorIndividual osobnik, Rezultat rezultat) {

    if (osobnik.getGene(pozycjaStartowaWGenotypie) == 1)
      return kodGraya.getWartoscKoduGraya(osobnik);    
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
    
    kodGraya.setValue(individual, (int)argumenty[1]);  
    
  }

}
