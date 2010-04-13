package pl.wroc.uni.ii.evolution.engine.operators.general.selections.fitness;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * @author Donata Malecka, Piotr Baraniak
 */
// TODO test it
public class EvMinimalizationFitness<T extends EvIndividual> extends
    EvIndividualFitness<T> {
  private double min = Double.MAX_VALUE;

  private double sum = 0.0;


  // to jest zle ...
  public double getFitness(T individual) {
    double fn_value;
    try {
      fn_value = Math.abs(individual.getObjectiveFunctionValue());

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
