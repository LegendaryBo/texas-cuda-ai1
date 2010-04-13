package pl.wroc.uni.ii.evolution.engine.operators.general.composition;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Applies a given operator on input population many times, till enough
 * individuals is generated to match the size of input population.<br>
 * Result population has exactly as many individuals as input population, if the
 * last operator's apply returned to many individuals, random individuals are
 * removed from population
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 */
public class EvIterateCompositon<T extends EvIndividual> implements
    EvOperator<T> {

  private EvOperator<T> operator;


  /**
   * Constructor
   * 
   * @param operator to be applied as many times as needed
   */
  public EvIterateCompositon(EvOperator<T> operator) {
    this.operator = operator;
  }


  /**
   * {@inheritDoc}
   */
  public EvPopulation<T> apply(EvPopulation<T> population) {
    int target_size = population.size();

    EvPopulation<T> children = new EvPopulation<T>();

    // apply operator till got enough individuals
    while (children.size() < target_size) {
      for (T i : operator.apply(population)) {
        children.add(i);
      }
    }

    // removing overflowing individuals
    while (children.size() > population.size()) {
      children.remove(EvRandomizer.INSTANCE.nextInt(children.size()));
    }

    return children;
  }

}
