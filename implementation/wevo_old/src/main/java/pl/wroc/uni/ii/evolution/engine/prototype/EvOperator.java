package pl.wroc.uni.ii.evolution.engine.prototype;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;

/**
 * Interface for all operators in wEvo.<br>
 * <br>
 * In wEvo, algorithms evaluate by running its operators, so if you want to
 * write custom algorithm, implement this interface and put your operator into
 * the EvAlgorithm object.
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 * @param <T> - type of individuals the operator works on
 */
public interface EvOperator<T extends EvIndividual> {

  // The operator should return new population of new individuals
  // (references to input individuals shouldn't be used)
  /**
   * Evaluates the operator on a given population and returns a new one.<br>
   * 
   * @param population a input population
   * @return output population
   */
  EvPopulation<T> apply(EvPopulation<T> population);
}
