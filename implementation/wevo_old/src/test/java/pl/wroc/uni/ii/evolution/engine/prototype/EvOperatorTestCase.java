package pl.wroc.uni.ii.evolution.engine.prototype;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperatorTestCase;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * This is a test for the operations of EvolutionaryOperator interface.
 * A concrete subclass should be derived for each non-abstract subclass
 * of EvolutionaryOperator.
 * 
 * @author Kamil Dworakowski
 *
 */
public abstract class EvOperatorTestCase<T extends EvIndividual> extends TestCase {
  
  /**
   * If all individuals in population have same objective function set then
   * all individuals in resulting population returned by method apply should
   * have the same function set.
   */
  public void testLeavingTheObjectiveFunctionOnIndividuals() throws Exception {
    EvOperator<T> operator = operatorUnderTest();
    EvPopulation<T> population = populationWithObjectiveFunctionSet();
    EvPopulation<T> returned_population = operator.apply(population);
    
    for (T indiv : returned_population)
      assertNotNull("individual should have objective function set", indiv.getObjectiveFunction());
  }

  
  /**
   * @return a population with a few individuals, they have to have a objective function set
   */
  protected abstract EvPopulation<T> populationWithObjectiveFunctionSet();
  
  /**
   * @return a operator to test
   */
  protected abstract EvOperator<T> operatorUnderTest();

}
