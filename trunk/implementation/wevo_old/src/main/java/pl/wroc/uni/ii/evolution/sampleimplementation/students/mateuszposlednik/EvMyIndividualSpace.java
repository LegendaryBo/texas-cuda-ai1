package pl.wroc.uni.ii.evolution.sampleimplementation.students.mateuszposlednik;

import java.util.Set;

import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;

/**
 * Zadanie rozgrzewkowe - przestrzen dla EvMyIndividual.
 * 
 * @author Mateusz Poslednik mateusz.poslednik@gmail.com
 */
public class EvMyIndividualSpace implements EvSolutionSpace<EvMyIndividual> {

  /** dfs. */
  private static final long serialVersionUID = -881856757980624787L;

  /** dlugosc chromosonu. */
  private final int length;

  /** funkcja celu. */
  private EvObjectiveFunction<EvMyIndividual> objectiveFunction;


  /**
   * Konstruktor.
   * 
   * @param objectiveFunction_ funkcja celu
   * @param length_ dlugosc chromosonu
   */
  public EvMyIndividualSpace(
      final EvObjectiveFunction<EvMyIndividual> objectiveFunction_,
      final int length_) {
    this.length = length_;
    this.objectiveFunction = objectiveFunction_;
  }


  /**
   * Sprzwdza czy osobnik pasuje do przestrzeni poszukiwan.
   * 
   * @param individual Osobnik do sprawdzenia
   * @return true gdy nalezy do danej przestrzeni
   */
  public boolean belongsTo(final EvMyIndividual individual) {
    if (individual == null) {
      return false;
    }
    return individual.getGenes().length == length;
  }


  /**
   * Dzieli przestrzen na kawalki.
   * 
   * @param n n
   * @return null
   */
  public Set<EvSolutionSpace<EvMyIndividual>> divide(final int n) {
    // nie obslugiwane
    return null;
  }


  /**
   * Dzieli przestrzen na kawalki.
   * 
   * @param n n
   * @param p p
   * @return null
   */
  public Set<EvSolutionSpace<EvMyIndividual>> divide(final int n,
      final Set<EvMyIndividual> p) {
    // nie obslugiwane
    return null;
  }


  /**
   * Tworzymy osobnika.
   * 
   * @return nowy osobnik
   */
  public EvMyIndividual generateIndividual() {
    EvMyIndividual in = new EvMyIndividual(length);
    in.setObjectiveFunction(this.objectiveFunction);
    return in;
  }


  /**
   * Zwraca funkcje celu.
   * 
   * @return funkcja celu
   */
  public EvObjectiveFunction<EvMyIndividual> getObjectiveFuntion() {
    return this.objectiveFunction;
  }


  /**
   * ustawia funkcjce celu.
   * 
   * @param objective_function funkcja celu
   */
  public void setObjectiveFuntion(
      final EvObjectiveFunction<EvMyIndividual> objective_function) {
    this.objectiveFunction = objective_function;

  }


  /**
   * Sprowadza osobnika, ktory jest poza przestrzenia poszukiwan. do
   * najblizszego w przestrzeni poszukiwan. Jesli dany osobnik jest w tej
   * przestrzeni to zwraca go.
   * 
   * @param individual Osobnik
   * @return Osobnik zgodny z przestrzenia poszukiwan
   */
  public EvMyIndividual takeBackTo(final EvMyIndividual individual) {
    if (this.belongsTo(individual)) {
      return individual;
    } else {
      return null;
    }
  }

}
