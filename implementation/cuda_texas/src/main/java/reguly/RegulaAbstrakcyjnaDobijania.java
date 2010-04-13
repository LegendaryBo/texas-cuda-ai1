package reguly;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import engine.Gra;
import engine.rezultaty.Rezultat;

public abstract class RegulaAbstrakcyjnaDobijania extends RegulaAbstrakcyjna {

  public RegulaAbstrakcyjnaDobijania(int pozycjaStartowaWGenotypie,
      int dlugoscReguly) {
    super(pozycjaStartowaWGenotypie, dlugoscReguly);
  }

  protected double stawka = 0.0d;
  
  @Override
  public abstract double aplikujRegule(Gra gra, int aKolejnosc, EvBinaryVectorIndividual osobnik, Rezultat rezultat);

  public double aplikujRegule(Gra aGra, int aKolejnosc, double aStawka, EvBinaryVectorIndividual osobnik, Rezultat rezultat) {
    stawka = aStawka;
    return aplikujRegule(aGra, aKolejnosc, osobnik, rezultat);
  }
  
}
