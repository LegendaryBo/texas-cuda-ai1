package pl.wroc.uni.ii.evolution.objectivefunctions;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvGenesSumFunction;
import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;

/**
 * Simple test checking if the function works
 * 
 * 
 * @author Kacper Gorski 'admin@34all.org'
 *
 */
public class EvGeneSumFunctionTest extends TestCase {
  
  public void testFunction() {
    int[] test_table = new int[] { 0,1,2 };
    EvKnaryIndividual test_individual = new EvKnaryIndividual(test_table, 3);
  
    test_individual.setObjectiveFunction(new EvGenesSumFunction());
    assertEquals(3.0, test_individual.getObjectiveFunctionValue());
    
    // optimum solution
    test_individual.setGene(0, 3);
    test_individual.setGene(1, 3);
    test_individual.setGene(2, 3);
    
    test_individual.setObjectiveFunction(new EvGenesSumFunction());
    assertEquals(9.0, test_individual.getObjectiveFunctionValue());    
  }
}
