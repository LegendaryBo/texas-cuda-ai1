package pl.wroc.uni.ii.evolution.engine.operators.general.selections.fitness;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * @author Marcin Golebiowski The class simply return objective function value
 *         of the individual. You can overwrite function getFitness to build
 *         more complex functions
 */
public class EvIndividualFitness<T extends EvIndividual> {

  protected EvPopulation<T> population;


  /**
   * Returns fitness of <code> individual </code> for a given
   * <code> population </code>.
   * 
   * @param individual
   * @param population
   * @return fitness
   */
  public double getFitness(T individual) {
    return individual.getObjectiveFunctionValue();
  }


  public void reinitialize(EvPopulation<T> population) {
    this.population = population;
  }
}
