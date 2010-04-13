package pl.wroc.uni.ii.evolution.engine.terminationconditions;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.conditions.EvTerminationCondition;

/**
 * Termination condition which is satisfied after changed it n-times.
 * 
 * @author Marek Chrusciel, Michal Humenczuk
 */
public class EvMaxIteration<T extends EvIndividual> implements
    EvTerminationCondition<T> {
  private int iteration = 0;

  private int max_iteration;


  /**
   * basic constructor
   * 
   * @param n maximum iterations number
   */
  public EvMaxIteration(int n) {
    max_iteration = n;
  }


  /**
   * Checks if iteration counter is at least equals fixed iterations number.
   */
  public boolean isSatisfied() {
    return iteration >= max_iteration;
  }


  /**
   * I ncrement iteration counter.
   */
  public void changeState(EvPopulation<T> parameters) {
    iteration++;
  }
}
