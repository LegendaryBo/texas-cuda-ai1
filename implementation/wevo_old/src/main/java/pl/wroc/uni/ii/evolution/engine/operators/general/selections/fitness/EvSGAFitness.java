package pl.wroc.uni.ii.evolution.engine.operators.general.selections.fitness;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * This class calculate fitness of individual from given population. It was
 * written for SGA algorithm. It works for positive objective function values.
 * 
 * @author Marcin Golebiowski, Piotr Baraniak
 */
public class EvSGAFitness<T extends EvIndividual> extends
    EvIndividualFitness<T> {

  private double min = Double.MAX_VALUE;

  private double sum = 0.0;


  /**
   * Returns fitness of <code> individual </code> to a given
   * <code> population </code>. Computes it as in SGA algorithm.
   * 
   * @param individual individual, which fitness we want to calculate
   * @param population individual is in it
   * @return fitness fitness for given individual
   */
  public double getFitness(T individual) {
    try {
      return (individual.getObjectiveFunctionValue() - min)
          / (sum - min * population.size());
    } catch (Exception e) {
      return 1;
    }
  }


  public void reinitialize(EvPopulation<T> population) {
    super.reinitialize(population);
    sum = 0.0;
    for (T i : population) {
      sum += i.getObjectiveFunctionValue();
      if (min > i.getObjectiveFunctionValue())
        min = i.getObjectiveFunctionValue();
    }
  }
}
