package pl.wroc.uni.ii.evolution.objectivefunctions.messy;

import java.util.ArrayList;
import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.messy.EvMessyBinaryVectorMaxSum;

/**
 * Simple test checking if class EvMessyBinaryMaxSum works
 * 
 * @author Piotr Staszak (stachhh@gmail.com)
 * @author Marek Szykula (marek.esz@gmail.com)
 */

public class EvMessyBinaryVectorMaxSumTest extends TestCase {

  // testing function on underspecified individual
  public void testUnderspecifiedIndividual() {
	  
    ArrayList<Integer> genes = new ArrayList<Integer>();
    ArrayList<Boolean> alleles = new ArrayList<Boolean>();
    
    genes.add(0);alleles.add(false);
    genes.add(1);alleles.add(true);
    genes.add(2);alleles.add(false);
    genes.add(2);alleles.add(true);
    genes.add(4);alleles.add(false);
    
    EvMessyBinaryVectorIndividual individual =
        new EvMessyBinaryVectorIndividual(5, genes, alleles); 
    EvMessyBinaryVectorMaxSum max_sum = new  EvMessyBinaryVectorMaxSum();
    
    assertEquals(max_sum.evaluate(individual), 2.0);
  }

  // testing function on individual with zero genes values only
  public void testZeroIndividual() {
	  
    ArrayList<Integer> genes = new ArrayList<Integer>();
    ArrayList<Boolean> alleles = new ArrayList<Boolean>();
    
    for (int i = 0; i < 5; i++) {
      genes.add(i);
      alleles.add(false);
    }
    
    EvMessyBinaryVectorIndividual individual =
        new EvMessyBinaryVectorIndividual(5, genes, alleles); 
    EvMessyBinaryVectorMaxSum max_sum = new  EvMessyBinaryVectorMaxSum();
    
    assertEquals(max_sum.evaluate(individual), 0.0);
  }
}