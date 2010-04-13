package pl.wroc.uni.ii.evolution.engine.prototype.conditions;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Basic interface that every TerminationCondition must implement. In
 * GenericEvolutionaryAlgorithm changeState is given a population as parameters.
 * In other implementations this may vary. This method should be called exactly
 * once each iteration.
 * 
 * @author Marcin Brodziak (marcin@nierobcietegowdomu.pl)
 * @author Tomasz Kozakiewicz (quzzaq@gmail.com)
 */
public interface EvTerminationCondition<T extends EvIndividual> {

  /**
   * Checks if termination condition is satisfied
   * 
   * @return
   */
  public boolean isSatisfied();


  /**
   * Change the state
   * 
   * @param parameters parameters for changing state
   */
  public void changeState(EvPopulation<T> parameters);
}
