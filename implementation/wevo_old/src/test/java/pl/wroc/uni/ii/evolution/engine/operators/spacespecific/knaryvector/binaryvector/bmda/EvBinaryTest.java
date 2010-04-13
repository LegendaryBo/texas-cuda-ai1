package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.bmda;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.bmda.bayesnetwork.EvBinary;
import junit.framework.TestCase;

/**
 * @author Jaroslaw Fuks (jarek102@gmail.com)
 * 
 * Simple test of class that counts number of zeros and ones in specified gene
 * and population.
 */
public class EvBinaryTest extends TestCase {

  /**
   * Primary test
   */
  public void testNumberOf() {
    EvPopulation<EvBinaryVectorIndividual> population = 
        new EvPopulation<EvBinaryVectorIndividual>();
    population.add(new EvBinaryVectorIndividual(new int[] {1, 1}));
    population.add(new EvBinaryVectorIndividual(new int[] {0, 1}));
    population.add(new EvBinaryVectorIndividual(new int[] {1, 1}));
    population.add(new EvBinaryVectorIndividual(new int[] {1, 1}));
    population.add(new EvBinaryVectorIndividual(new int[] {0, 1}));
    population.add(new EvBinaryVectorIndividual(new int[] {0, 0}));
    population.add(new EvBinaryVectorIndividual(new int[] {0, 0}));
    
    assertEquals(3, EvBinary.numberOf(population,
        new int[] {1, 1}, new int[] {0, 1}));
    
    assertEquals(0, EvBinary.numberOf(population,
        new int[] {1, 0}, new int[] {0, 1}));
    
    
    assertEquals(2, EvBinary.numberOf(population,
        new int[] {0, 0}, new int[] {0, 1}));
    
   
    
  }

  /**
   * ?
   */
  public void testCompare() {
    
    assertTrue(EvBinary.compare(
        new EvBinaryVectorIndividual(new int[] {1, 1, 0}),
        
            new int[] {1}, new int[] {0}));
    
    assertTrue(EvBinary.compare(
        new EvBinaryVectorIndividual(new int[] {1, 1, 0}),
        
            new int[] {1, 0}, new int[] {0, 2}));
        
    assertFalse(EvBinary.compare(
        new EvBinaryVectorIndividual(new int[] {1, 1, 0}),
        
            new int[] {1, 1, 0}, new int[] {0, 2, 1}));
        
    assertTrue(EvBinary.compare(
        new EvBinaryVectorIndividual(new int[] {1, 0, 0}),
        
            new int[] {1, 0, 0}, new int[] {0, 2, 1}));
        
    
  }

  
  /**
   * Testing of function converting bool to int table.
   */
  public void testIntToBools() {
    
  
    
    
    int[] pattern1 = new int[] {1, 1};
    int[] result1 = EvBinary.intToBools(3, 2);
    assertEquals(pattern1.length, result1.length);
    assertEquals(pattern1[0], result1[0]);
    assertEquals(pattern1[1], result1[1]);    
    
    int[] pattern2 = new int[] {1, 0, 1};
    int[] result2 = EvBinary.intToBools(5, 3);
    assertEquals(pattern2.length, result2.length);
    assertEquals(pattern2[0], result2[0]);
    assertEquals(pattern2[1], result2[1]);   
    assertEquals(pattern2[2], result2[2]);  
    
    
    
    int[] pattern3 = new int[] {1};
    int[] result3 = EvBinary.intToBools(5, 1);
    assertEquals(pattern2.length, result2.length);
    assertEquals(pattern3[0], result3[0]);
    
    
  }

  /**
   * Testing ^2 function.
   */
  public void testPow2() {
    
    for (int i = 0; i < 31; i++) {
     assertEquals(Math.pow(2, i), (double) EvBinary.pow2(i));
    }
  }

  /**
   * Testing factorial function.
   */
  public void testFactorial() {
    assertEquals(1.0,EvBinary.factorial(1));
    assertEquals(1.0, EvBinary.factorial(0));
    assertEquals(2.0, EvBinary.factorial(2));
    assertEquals(6.0, EvBinary.factorial(3));
    assertEquals(24.0, EvBinary.factorial(4));
    
  }

  /**
   *  Testing function that summs all log1 + log2 + ... + logn.
   */
  public void testSumLog() {
    assertEquals(Math.log(1) + Math.log(2), EvBinary.sumLog(2));
    assertEquals(0.0, EvBinary.sumLog(0));
    
  }

}
