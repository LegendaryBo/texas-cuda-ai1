package reguly;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import engine.Gra;
import engine.rezultaty.Rezultat;

/**
 * 
 * Abstrakcyjna klasa dla regul
 * 
 * @author KAcper Gorski
 *
 */
public abstract class RegulaAbstrakcyjna {
  
  // pozycja reguly w genotypie (index pierwszego genu)
  protected final int pozycjaStartowaWGenotypie;
  
  // dlugosc reguly w genotypie
  protected final int dlugoscReguly;
  

  
  public RegulaAbstrakcyjna(int aPozycjaStartowaWGenotypie, int aDlugoscReguly) {
    pozycjaStartowaWGenotypie = aPozycjaStartowaWGenotypie;  
    dlugoscReguly = aDlugoscReguly;

  }
  
  
  /**
   * Ta metoda zwraca decyzje reguly
   * 
   * @param gra
   * @param kolejnosc 
   * @return
   */
  public abstract double aplikujRegule(Gra gra, int kolejnosc, EvBinaryVectorIndividual aOsobnik, Rezultat rezultat);
  
  
  public abstract void zmienIndividuala(double[] argumenty, EvBinaryVectorIndividual individual);
  
  
  public int getPozycjaStartowaWGenotypie() {
    return pozycjaStartowaWGenotypie;
  }
  
  public int getDlugoscReguly() {
    return dlugoscReguly;
  }
  
}
