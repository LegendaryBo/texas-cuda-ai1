package pl.wroc.uni.ii.evolution.engine.operators.general.composition;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Operator for composing 2 other operators.<br>
 * It applies first operator to the input population, then applies second
 * operator to the results.
 * 
 * @author Jarek Fuks (jarek102@gmail.com)
 * @author Marcin Brodziak (marcin@nierobcietegowdomu.com)
 */
public class EvTwoOperatorsComposition<T extends EvIndividual> implements
    EvOperator<T> {
  private EvOperator<T> first, second;


  /**
   * Constructor
   * 
   * @param later operator that is applied after earlier operator
   * @param earlier operator that is first applied on population
   */
  public EvTwoOperatorsComposition(EvOperator<T> later, EvOperator<T> earlier) {
    this.first = later;
    this.second = earlier;
  }


  /**
   * {@inheritDoc}
   */
  public EvPopulation<T> apply(EvPopulation<T> population) {
    return first.apply(second.apply(population));
  }

}
