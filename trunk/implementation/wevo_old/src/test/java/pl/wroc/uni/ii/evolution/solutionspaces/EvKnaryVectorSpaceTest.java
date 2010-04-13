package pl.wroc.uni.ii.evolution.solutionspaces;

import pl.wroc.uni.ii.evolution.engine.EvGenesSumFunction;
import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import junit.framework.TestCase;

/**
 * tests all implemented functions and constructors
 * 
 * @author Kacper Gorski (admin@34all.org)
 *
 */
public class EvKnaryVectorSpaceTest extends TestCase {

  // contructor, getDimension, objective functions
  public void testBasics() {
    int dimension = 10;
    int max_gene_value = 3;
    
    EvKnaryVectorSpace knary_space = 
      new EvKnaryVectorSpace(dimension, max_gene_value);
    
    assertEquals(dimension, knary_space.getDimension());
    
    EvGenesSumFunction function = new EvGenesSumFunction();
    knary_space.setObjectiveFuntion(function);
    
    assertEquals(function, knary_space.getObjectiveFuntion());
    
  }
  
  // test if generated individuals are ok
  public void testGenerateIndividuals() {
    
    EvKnaryVectorSpace knary_solution = new EvKnaryVectorSpace(20,5);
    EvKnaryIndividual individual = knary_solution.generateIndividual();
    
    // check if generated individual has different genes than blank individual
    assertNotSame((new int[20]).hashCode(), individual.getGenes().hashCode());
    
    // check if generated individual has correct class
    assertEquals(individual.getClass(), EvKnaryIndividual.class);
    
  }
  
}
