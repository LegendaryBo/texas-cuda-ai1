package reguly.LicytacjaNaWejscie;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.RegulaAbstrakcyjna;
import reguly.kodGraya.KodGraya;
import engine.Gra;
import engine.rezultaty.Rezultat;

/**
 * 
 * Regula zwraca ilosc 'glosow', jesli spelniony jest warunek:
 * wartosc puli jest mniejsza od zdefiniowanej w genotypie
 * 
 * @author Kacper Gorski
 *
 */
public class RegulaOgraniczenieStawki extends RegulaAbstrakcyjna {

  public KodGraya kodGrayaWagaGlosu = null; 
  public KodGraya kodGrayaOgranicznik = null; 
  
  public RegulaOgraniczenieStawki(int pozycjaStartowaWGenotypie, int aDlugoscKoduGrayaWagi, int aDlugoscKoduGrayaOgranicznik) {
    super(pozycjaStartowaWGenotypie, 1 + aDlugoscKoduGrayaWagi + aDlugoscKoduGrayaOgranicznik);
    
    kodGrayaWagaGlosu = new KodGraya(aDlugoscKoduGrayaWagi ,pozycjaStartowaWGenotypie + 1);
    
    kodGrayaOgranicznik = new KodGraya(aDlugoscKoduGrayaOgranicznik ,pozycjaStartowaWGenotypie + 1 +  aDlugoscKoduGrayaWagi);
  }

  @Override
  public double aplikujRegule(Gra gra, int kolejnosc, EvBinaryVectorIndividual osobnik, Rezultat rezultat) {
	  
//	  System.out.println(gra.stawka);
	  
 // jesli regula jest wlaczona
    if (osobnik.getGene(pozycjaStartowaWGenotypie) == 1 ) {

      if (gra.stawka <= kodGrayaOgranicznik.getWartoscKoduGraya(osobnik)) {
        return (double)kodGrayaWagaGlosu.getWartoscKoduGraya(osobnik); // to zwroc 'waznosc' tej reguly
      }
      else {
        return 0.0d;
      }
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
    
    kodGrayaWagaGlosu.setValue(individual, (int)argumenty[1]);  
    kodGrayaOgranicznik.setValue(individual, (int)argumenty[2]);  
  }

}
