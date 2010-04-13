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
 * @param <T> - type of individuals the mutation operates on
 */
public abstract class EvMutation<T extends EvIndividual> implements
    EvOperator<T> {

  private boolean mutate_clone = false;


  @SuppressWarnings("unchecked")
  public EvPopulation<T> apply(EvPopulation<T> population) {

    EvPopulation<T> result = new EvPopulation<T>();

    for (T individual : population) {
      if (mutate_clone) {
        result.add(mutate((T) individual.clone()));
      } else {
        result.add(mutate(individual));
      }
    }

    return result;
  }


  /**
   * Mutates an individual
   * 
   * @param individual a given individual to mutate
   * @return mutated individual, it should be the given individual with
   *         modifications, not new individual
   */
  public abstract T mutate(T individual);


  /**
   * Sets if mutation is done on clone of given individual or not
   * 
   * @param mutate_clone
   */
  public void setMutateClone(boolean mutate_clone) {
    this.mutate_clone = mutate_clone;
  }

}
