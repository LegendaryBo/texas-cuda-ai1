package pl.wroc.uni.ii.evolution.engine.operators.general.composition;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * It returns new population by adding individuals created by specified operator
 * to the old population.<br>
 * <br>
 * EXAMPLE:<br>
 * we use EvAddResultComposition with EvKnaryOnePointCrossover on population:<br>
 * (0,0,0,0,0,0) ; (1,1,1,1,1,1)<br>
 * <br>
 * EvKnaryOnePointCrossover created 2 new individuals:<br>
 * (0,0,0,1,1,1) ; (1,1,1,0,0,0)<br>
 * <br>
 * Output population will be:<BR>
 * (0,0,0,0,0,0) ; (1,1,1,1,1,1) ; (0,0,0,1,1,1) ; (1,1,1,0,0,0)<br>
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 * @param <T> - type of individuals the operator works on
 */
public class EvAddResultComposition<T extends EvIndividual> implements
    EvOperator<T> {

  private EvOperator<T> operator;


  /**
   * Construction
   * 
   * @param operator operator shouldn't modify an input population
   */
  public EvAddResultComposition(EvOperator<T> operator) {
    this.operator = operator;
  }


  /**
   * {@inheritDoc}
   */
  public EvPopulation<T> apply(EvPopulation<T> population) {

    EvPopulation<T> result = operator.apply(population);
    for (T ind : population) {
      result.add(ind);
    }

    return result;
  }

}
