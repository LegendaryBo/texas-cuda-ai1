package reguly;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.kodGraya.KodGraya;
import engine.Gra;
import engine.rezultaty.Rezultat;

/**
 * 
 * Abstrakcyjna regula kodu graya.
 * 
 * - mozna ja wlaczyc/wylaczyc
 * - zdefiniowac jej wage
 * 
 * @author Kacper Gorski
 *
 */
public abstract class RegulaAbstrakcyjnaIleGrac extends RegulaAbstrakcyjna {

  public KodGraya kodGrayaWagi = null; 
  private int dlugoscWagi;
  
  public int getDlugoscWagi() {
    return dlugoscWagi;
  }
  
  public RegulaAbstrakcyjnaIleGrac(int pozycjaStartowaWGenotypie,
      int dlugoscReguly, int dlugosc_wagi) {
    super(pozycjaStartowaWGenotypie, dlugoscReguly);
    this.dlugoscWagi = dlugosc_wagi;
    
    kodGrayaWagi = new KodGraya(dlugosc_wagi, pozycjaStartowaWGenotypie+1);
    
  }

  public float[] aplikRegule(Gra gra, int kolejnosc,
      EvBinaryVectorIndividual osobnik, Rezultat rezultat, float stawka) {
    float[] wynik = new float[2];
    
    wynik[0] = (float) aplikujRegule(gra, kolejnosc, osobnik, rezultat);
    wynik[1] = kodGrayaWagi.getWartoscKoduGraya(osobnik);
    
    return wynik;
  }
  


  @Override
  public void zmienIndividuala(double[] argumenty,
      EvBinaryVectorIndividual individual) {

  }

}
