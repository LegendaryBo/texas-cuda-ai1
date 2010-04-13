package pl.wroc.uni.ii.evolution.sampleimplementation.students.mateuszposlednik;

import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

/**
 * Funkcja celu dla zadania rozgrzewkowego. Opis: Geny sa liczbami calkowitymi.
 * Jesli numer pozycji genu jest podzielny przez dwa to wartosc tego genu
 * dodajemy do wyniku. W przeciwnym wypadku odejmujemy. Np.: geny =
 * {0,1,2,3,4,5}; wartosc = 0 - 1 + 2 - 3 + 4 - 5 Np.2: geny = {3,4,5,1,2}
 * wartosc = 3 - 4 + 5 - 1 + 2
 * 
 * @author Mateusz Poslednik mateusz.poslednik@gmail.com
 */
public class EvMyIndividualObjective implements
    EvObjectiveFunction<EvMyIndividual> {

  /**
   * 
   */
  private static final long serialVersionUID = -5119573631043413108L;


  /**
   * Konstruktor.
   */
  public EvMyIndividualObjective() {

  }


  /**
   * Liczymy wartosc osobnika dla tej funkcji celu.
   * 
   * @param individual Osobnik do ocenienia
   * @return Wartosc osobnika
   */
  public double evaluate(final EvMyIndividual individual) {
    int[] genes = individual.getGenes();
    double value = 0;
    for (int i = 0; i < genes.length; i++) {
      if (i % 2 == 0) {
        value += genes[i];
      } else {
        value -= genes[i];
      }
    }
    return value;
  }

}
