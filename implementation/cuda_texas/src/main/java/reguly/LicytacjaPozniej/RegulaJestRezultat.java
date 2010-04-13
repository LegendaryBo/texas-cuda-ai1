package reguly.LicytacjaPozniej;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.RegulaAbstrakcyjna;
import reguly.kodGraya.KodGraya;
import engine.Gra;
import engine.rezultaty.Rezultat;

/**
 * 
 * Regula zwraca stawke zakodowana w kodzie graya, jesli gracz posiada podany wynik
 * 
 * @author Kacper Gorski
 *
 */
public class RegulaJestRezultat extends RegulaAbstrakcyjna {

  private KodGraya kodGraya = null;
  private byte rezultat;
  
  // rezultat:
  // smiec = 1
  // para = 2
  // 2 pary = 3
  // trojka = 4
  // street = 5
  // kolor = 6
  // full = 7
  // kareta = 8
  // poker = 9
  public RegulaJestRezultat(int pozycjaStartowaWGenotypie, int dlugoscReguly, int aRezultat) {
    super(pozycjaStartowaWGenotypie, dlugoscReguly +1);
    rezultat = (byte) aRezultat;
    kodGraya = new KodGraya(dlugoscReguly, pozycjaStartowaWGenotypie+1);
  }

  @Override
  public double aplikujRegule(Gra aGra, int aKolejnosc, EvBinaryVectorIndividual osobnik, Rezultat aRezultat) {


    if (osobnik.getGene(pozycjaStartowaWGenotypie) == 1 ) {
  
      // jesli regula jest wlaczona
      if (aRezultat.poziom >= rezultat)
        return (double)kodGraya.getWartoscKoduGraya(osobnik); 
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
    
    rezultat = (byte) argumenty[1];
    
    kodGraya.setValue(individual, (int)argumenty[2]);
    
  }

}
