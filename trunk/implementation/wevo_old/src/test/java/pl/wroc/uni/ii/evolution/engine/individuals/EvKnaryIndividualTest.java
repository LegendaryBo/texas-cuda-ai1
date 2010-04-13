package pl.wroc.uni.ii.evolution.engine.individuals;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvGenesSumFunction;

/**
 * 
 * Simple tests to check, if all methods and constructors
 * in EvKnaryIndividual class works well
 * 
 * @author Kacper Gorski 'admin@34all.org'
 *
 */
public class EvKnaryIndividualTest  extends TestCase {

  public void testConstructor() {
    
    final int test_length = 50;
    // test if genes are set to 0 by default
    EvKnaryIndividual test_individual = new EvKnaryIndividual(test_length, 3);
    
    assertTrue( test_individual.getGene(5) == 0);
    
    assertTrue( test_individual.getDimension() == 50);
    
    test_individual.setObjectiveFunction(new EvGenesSumFunction());
    
    assertEquals(0.0, test_individual.getObjectiveFunctionValue() );
    
  }
  
  // testing contructor with table of integers as parameter
  public void testConstructorFromIntTable() {
    
    int[] test_table = new int[] { 0,1,2 };
    EvKnaryIndividual test_individual = new EvKnaryIndividual(test_table, 3);
    
    assertEquals(test_individual.getDimension(), 3);
    
    assertEquals(test_individual.getGene(2), 2);
    
    test_individual.setObjectiveFunction(new EvGenesSumFunction());
    assertEquals(3.0, test_individual.getObjectiveFunctionValue() );
    
    // check if test_table is as different object than individuals genes' table
    test_table[0] = 2;
    assertEquals(test_individual.getGene(0), 0);
    
  }
  
  public void testSetGene() {
    
    int[] test_table = new int[] { 4,2,1 };
    
    EvKnaryIndividual test_individual = new EvKnaryIndividual(test_table, 5);   
    
    test_individual.setGene(2, 5);
    
    assertEquals(test_individual.getGene(2), 5);
    assertEquals(test_individual.getGene(1), 2);
    
  }
  
  public void testCloneHashAndEquals() {

    int[] test_table = new int[] { 4,2,1 };
    
    EvKnaryIndividual test_individual = new EvKnaryIndividual(test_table, 5);      
    EvKnaryIndividual test_individual2 = test_individual.clone();
    
    assertTrue( test_individual.hashCode() == test_individual2.hashCode());
    assertTrue( test_individual != test_individual2);
    assertTrue( test_individual.equals(test_individual2) );
    assertTrue( test_individual2.equals(test_individual) );
    
    test_individual.setGene(0, 0);
    
    // check if genes tables are different
    assertEquals(test_individual2.getGene(0), 4);
    assertTrue( test_individual.hashCode() != test_individual2.hashCode());
    assertTrue( !test_individual.equals(test_individual2) );
    assertTrue( !test_individual2.equals(test_individual) );    
  }
  
  public void testToString() {

    int[] test_table = new int[] { 4,2,1 };
    
    EvKnaryIndividual test_individual = new EvKnaryIndividual(test_table, 5);   
    
    assertEquals("421", test_individual.toString());
    
  }
  
  // check if individual notices when it's objective function value
  // should be evaluated again
  public void testIvalidate() {

    int[] test_table = new int[] { 0,1,2 };
    EvKnaryIndividual test_individual = new EvKnaryIndividual(test_table, 3);
    
    assertTrue(!test_individual.isObjectiveFunctionValueCalculated());
    
    EvGenesSumFunction obj_function = new EvGenesSumFunction();
    test_individual.setObjectiveFunction(obj_function);
    
    assertTrue(! test_individual.isObjectiveFunctionValueCalculated() );
    
    test_individual.getObjectiveFunctionValue();  
    assertTrue( test_individual.isObjectiveFunctionValueCalculated());
    
    test_individual.setGene(2, 0);
    assertTrue(! test_individual.isObjectiveFunctionValueCalculated());
    
  }
  
}
