package pl.wroc.uni.ii.evolution.engine.terminationconditions;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.conditions.EvTerminationCondition;

/**
 * Termination condition which is satisfied when best value in population did
 * not change for k iterations.
 * 
 * @author Marcin Brodziak
 */
public class EvBestValueNotImproved<T extends EvIndividual> implements
    EvTerminationCondition<T> {
  private double current_best = Double.NEGATIVE_INFINITY;

  private int iterations_with_no_change = 0;

  private final int max_iterations_with_no_change;


  /**
   * basic constructor
   * 
   * @param n maximum iterations with no change
   */
  public EvBestValueNotImproved(int n) {
    max_iterations_with_no_change = n;
  }


  /**
   * Checks if iteration counter has not exceeded the maximum
   */
  public boolean isSatisfied() {
    return iterations_with_no_change > max_iterations_with_no_change;
  }


  /**
   * Increment iteration counter.
   */
  public void changeState(EvPopulation<T> parameters) {
    if (parameters.getBestResult().getObjectiveFunctionValue() > current_best) {
      current_best = parameters.getBestResult().getObjectiveFunctionValue();
      iterations_with_no_change = 0;
    } else {
      iterations_with_no_change++;
    }
  }
}
