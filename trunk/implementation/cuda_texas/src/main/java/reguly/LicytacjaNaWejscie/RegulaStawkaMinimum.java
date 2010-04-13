package reguly.LicytacjaNaWejscie;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.RegulaAbstrakcyjna;
import reguly.kodGraya.KodGraya;
import engine.Gra;
import engine.rezultaty.Rezultat;

/**
 * 
 * Regula zwraca ilosc 'glosow', jesli spelniony jest warunek:
 * w prywatnej rece jest PARA kart o tym samym kolorze
 * 
 * @author Kacper Gorski
 *
 */
public class RegulaStawkaMinimum extends RegulaAbstrakcyjna {

  private KodGraya kodGraya = null; 
  
  public RegulaStawkaMinimum(int pozycjaStartowaWGenotypie, int aDlugoscKoduGraya) {
    super(pozycjaStartowaWGenotypie, 1 + aDlugoscKoduGraya);
    
    kodGraya = new KodGraya(aDlugoscKoduGraya ,pozycjaStartowaWGenotypie + 1);
    
  }

  @Override
  public double aplikujRegule(Gra gra, int kolejnosc, EvBinaryVectorIndividual osobnik, Rezultat rezultat) {
  
    // jesli jest para
    if (gra.stawka == gra.minimal_bid) {
  
      // jesli regula jest wlaczona
      if (osobnik.getGene(pozycjaStartowaWGenotypie) == 1 )
        return (double)kodGraya.getWartoscKoduGraya(osobnik); // to zwroc 'waznosc' tej reguly
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
    
    kodGraya.setValue(individual, (int)argumenty[1]);  
    
  }

}