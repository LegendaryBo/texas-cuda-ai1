package pl.wroc.uni.ii.evolution.engine.operators.general.selections.fitness;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * This class calculate fitness of individual from given population. It is good
 * for objective functions which optimum is zero.
 * 
 * @author Piotr Baraniak, Lukasz Witko
 * @param <T>
 */
public class EvNearZeroFitness<T extends EvIndividual> extends
    EvIndividualFitness<T> {
  private double min = Double.MAX_VALUE;

  private double sum = 0.0;


  /**
   * Returns fitness of <code> individual </code> to a given
   * <code> population </code>. Computed fitness is as good as
   * getObjectiveFunctionValue is close to zero.
   * 
   * @param individual individual, which fitness we want to calculate
   * @param population individual is in it
   * @return fitness fitness for given individual
   */
  public double getFitness(T individual) {
    double fn_value;
    try {
      fn_value = Math.abs(individual.getObjectiveFunctionValue());

      if (fn_value == 0)
        fn_value = Double.MAX_VALUE;
      else
        fn_value = 1 / fn_value;

      fn_value /= population.size();
      return (fn_value - min) / (sum - min * population.size());
    } catch (Exception e) {
      return 1;
    }
  }


  public void reinitialize(EvPopulation<T> population) {
    super.reinitialize(population);
    double fn_value;
    sum = 0.0;
    for (T i : population) {
      fn_value = Math.abs(i.getObjectiveFunctionValue());
      if (fn_value == 0)
        fn_value = Double.MAX_VALUE;
      else
        fn_value = 1 / fn_value;

      fn_value /= population.size();

      sum += fn_value;

      if (min > fn_value)
        min = fn_value;
    }
  }

}
