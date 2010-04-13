package pl.wroc.uni.ii.evolution.sampleimplementation.students.mateuszposlednik;

import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Klasa utworzona w ramach zadania rozgrzewkowego. Osobnik ma chromoson
 * skladajacy sie z cyfr naturalnych od 1 do dl. chromosonu.
 * 
 * @author Mateusz Poslednik mateusz.poslednik@gmail.com
 */
public class EvMyIndividual extends EvIndividual {

  /**
   * 
   */
  private static final long serialVersionUID = 3141346064910753929L;

  /** geny osobnika. */
  private int[] genes;


  /**
   * Konstruktor tworzoacy osobnika o zadanej dlugosci.
   * 
   * @param length dlugosc chromosonu.
   */
  public EvMyIndividual(final int length) {
    genes = new int[length];
    for (int i = 1; i <= length; i++) {
      genes[i - 1] = i;
    }
  }


  /**
   * Konstruktor tworzocy osobnika na podstawie genow.
   * 
   * @param genes_ geny.
   */
  public EvMyIndividual(final int[] genes_) {
    this.genes = new int[genes_.length];
    for (int i = 0; i < genes_.length; i++) {
      this.genes[i] = genes_[i];
    }
  }


  /**
   * Klonuje osobnika.
   * 
   * @return Skopiowany osobnik.
   */
  @Override
  public EvMyIndividual clone() {
    EvMyIndividual copy = new EvMyIndividual(this.genes);
    return copy;
  }


  /**
   * Zwraca geny.
   * 
   * @return geny
   */
  public int[] getGenes() {
    return genes;
  }


  /**
   * Ustawia geny.
   * 
   * @param genes_ geny
   */
  public void setGenes(final int[] genes_) {
    this.genes = genes_;
  }

}
