package reguly.ileGrac;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.RegulaAbstrakcyjnaIleGrac;
import reguly.kodGraya.KodGraya;
import engine.Gra;
import engine.rezultaty.Rezultat;

public class IleGracStawkaRX extends RegulaAbstrakcyjnaIleGrac {

 // FI - wspolczynnik 0-1, ktory mnozy sie przez maksymalna mozliwa stawke
  
  public KodGraya gray_fi;
  private final int DLUGOSC_FI=5;
  private final int WIELKOSC_FI=31;
  public KodGraya gray_stawka;
  
  public IleGracStawkaRX(int pozycjaStartowaWGenotypie, int dlugosc_wagi, int dlugosc_stawka) {
    super(pozycjaStartowaWGenotypie, 1+dlugosc_wagi+5+dlugosc_stawka, dlugosc_wagi);
    
    gray_fi = new KodGraya(DLUGOSC_FI, pozycjaStartowaWGenotypie + dlugosc_wagi + 1);
    gray_stawka = new KodGraya(dlugosc_stawka, pozycjaStartowaWGenotypie + dlugosc_wagi + 1 + DLUGOSC_FI);
    
  }

  @Override
  public double aplikujRegule(Gra gra, int kolejnosc,
      EvBinaryVectorIndividual osobnik, Rezultat rezultat) {

    if ( osobnik.getGene(pozycjaStartowaWGenotypie) == 1 ) {   
      if (gra.stawka >= 10 * gra.minimal_bid * gray_stawka.getWartoscKoduGraya(osobnik))
        return (2.0d / WIELKOSC_FI) * gray_fi.getWartoscKoduGraya(osobnik);
      else 
        return -1.0d; 
    }
    else
      return -1.0d;
    
  }

}
