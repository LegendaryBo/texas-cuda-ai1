package reguly.LicytacjaNaWejscie;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.RegulaAbstrakcyjna;
import reguly.kodGraya.KodGraya;
import engine.Gra;
import engine.rezultaty.Rezultat;

/**
 * 
 * Zwraca, ile jest wymaganych glosow, aby wejsc do gry
 * 
 * @author Kacper Gorski
 *
 */
public class RegulaWymaganychGlosow extends RegulaAbstrakcyjna {

  public KodGraya kodGraya = null; 
  
  public RegulaWymaganychGlosow(int pozycjaStartowaWGenotypie, int aDlugoscKoduGraya) {
    super(pozycjaStartowaWGenotypie, aDlugoscKoduGraya);
    
    kodGraya = new KodGraya(aDlugoscKoduGraya ,pozycjaStartowaWGenotypie);
    
  }

  @Override
  public double aplikujRegule(Gra gra, int kolejnosc, EvBinaryVectorIndividual osobnik, Rezultat rezultat) {

    return kodGraya.getWartoscKoduGraya(osobnik);    

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
