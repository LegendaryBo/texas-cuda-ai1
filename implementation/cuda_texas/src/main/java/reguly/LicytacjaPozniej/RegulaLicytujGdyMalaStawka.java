package reguly.LicytacjaPozniej;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.RegulaAbstrakcyjna;
import reguly.kodGraya.KodGraya;
import engine.Gra;
import engine.rezultaty.Rezultat;

/**
 * 
 * 
 * @author Kacper Gorski
 *
 */
public class RegulaLicytujGdyMalaStawka extends RegulaAbstrakcyjna {

  private KodGraya kodGrayaStawka = null; 
  private KodGraya kodGrayaOgranicznik = null; 
  
  public RegulaLicytujGdyMalaStawka(int pozycjaStartowaWGenotypie,
      int aDlugoscKoduGrayaWagi, int aDlugoscKoduGrayaOgranicznik) {
    super(pozycjaStartowaWGenotypie, 1 + aDlugoscKoduGrayaWagi + aDlugoscKoduGrayaOgranicznik);
    
    kodGrayaStawka = new KodGraya(aDlugoscKoduGrayaWagi ,pozycjaStartowaWGenotypie + 1);
    
    kodGrayaOgranicznik = new KodGraya(aDlugoscKoduGrayaOgranicznik ,pozycjaStartowaWGenotypie + 1 +  aDlugoscKoduGrayaWagi);
  }

  @Override
  public double aplikujRegule(Gra gra, int kolejnosc, EvBinaryVectorIndividual osobnik, Rezultat rezultat) {
  
    // jesli jest para
    if (gra.stawka <=  kodGrayaOgranicznik.getWartoscKoduGraya(osobnik) * gra.minimal_bid) {
  
      // jesli regula jest wlaczona
      if (osobnik.getGene(pozycjaStartowaWGenotypie) == 1 )
        return (double)kodGrayaStawka.getWartoscKoduGraya(osobnik); // to zwroc 'waznosc' tej reguly
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
    
    kodGrayaStawka.setValue(individual, (int)argumenty[1]);
    
    kodGrayaOgranicznik.setValue(individual, (int)argumenty[2]);
    
  }

}
