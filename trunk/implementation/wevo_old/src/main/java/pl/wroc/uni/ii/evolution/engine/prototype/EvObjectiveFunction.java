package pl.wroc.uni.ii.evolution.engine.prototype;

import java.io.Serializable;

/**
 * Interface that every ObjectiveFunction must implement. getFitness method
 * should do basic validation of parameter to check if it can handle that kind
 * of individual. Important: Better individuals have higher objective values!
 * 
 * @author Marcin Brodziak (marcin@nierobcietegowdomu.pl)
 * @author Tomasz Kozakiewicz (quzzaq@gmail.com)
 * @param <T> - type of individuals the operator works on
 */
public interface EvObjectiveFunction<T extends EvIndividual> extends
    Serializable {

  /**
   * Evaluates an individual.
   * 
   * @param individual -- some individual to evaluate
   * @return individual's objective value
   */
  double evaluate(T individual);

}
