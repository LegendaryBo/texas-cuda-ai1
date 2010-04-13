/**
 * 
 */
package pl.wroc.uni.ii.evolution.engine.samplealgorithms;

// off Regula
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryUMDAProbability;

// on Regula

/**
 * UMDA Algorithm.
 * 
 * @author Adam Palka
 */
public final class EvUMDA extends EvAlgorithm<EvBinaryVectorIndividual> {

  /** Number of individuals returned by selection. */
  private int bestIndividualsNumber;


  /**
   * UMDA constructor.
   * 
   * @param population_size population size
   * @param best_individuals number of best individuals taked tn calculate
   *        probability
   */
  public EvUMDA(final int population_size, final int best_individuals) {
    super(population_size);
    this.bestIndividualsNumber = best_individuals;
  }


  /**
   * Behavior on initiation.
   */
  public void init() {
    super.init();

    super.addOperator(new EvKBestSelection<EvBinaryVectorIndividual>(
        bestIndividualsNumber));
    super.addOperator(new EvBinaryUMDAProbability(this.population_size));
  }


  /**
   * Number of individual returned by selection.
   * 
   * @return bestIndividualsNumber number of best individuals
   */
  public int getBestIndividualsNumber() {
    return bestIndividualsNumber;
  }


  /**
   * Changes number of individual returned by selection.
   * 
   * @param best_individuals number of best individuals
   */
  public void setBestIndividualsNumber(final int best_individuals) {
    this.bestIndividualsNumber = best_individuals;
  }
}
