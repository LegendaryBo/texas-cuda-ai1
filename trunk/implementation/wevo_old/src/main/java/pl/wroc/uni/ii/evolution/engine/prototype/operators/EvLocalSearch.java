package pl.wroc.uni.ii.evolution.engine.prototype.operators;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * An abstract class for all mutation operators. <br>
 * A mutation operator returns a changed populations. <br>
 * Some mutation is applied on every individual.
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 * @param <T> - type of individuals the local search operates on
 */
public abstract class EvLocalSearch<T extends EvIndividual> implements
    EvOperator<T> {

  public EvPopulation<T> apply(EvPopulation<T> population) {

    EvPopulation<T> result = new EvPopulation<T>();

    for (T individual : population) {
      result.add(search(individual));
    }

    return result;
  }


  /**
   * Search a other, maybe better, individual using given.
   * 
   * @param individual
   * @return found individual
   */
  public abstract T search(T individual);
}