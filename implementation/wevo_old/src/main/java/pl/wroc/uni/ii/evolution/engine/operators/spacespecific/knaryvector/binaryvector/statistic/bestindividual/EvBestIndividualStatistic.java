package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.statistic.bestindividual;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;

/**
 * @author Tomasz Kozakiewicz
 */
public class EvBestIndividualStatistic extends EvStatistic {
  // TODO generalize or change name
  private static final long serialVersionUID = -5435220480835060782L;

  private int[] best_individual;

  private double objective_function_value;


  public EvBestIndividualStatistic(int[] best_individual,
      double objective_function_value, int iteration) {
    this.best_individual = best_individual;
    this.objective_function_value = objective_function_value;
    setIteration(iteration);
  }


  public int[] getBits() {
    return best_individual;
  }


  public double getObjectiveFunctionValue() {
    return objective_function_value;
  }
}
