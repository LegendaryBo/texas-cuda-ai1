package pl.wroc.uni.ii.evolution.objectivefunctions.messy;

import java.util.ArrayList;
import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.objectivefunctions.messy.EvMessyBinaryVectorObjectiveFunctionWrapper;

/**
 * Simple test checking if the wrapper works
 * 
 * @author Piotr Staszak (stachhh@gmail.com)
 * @author Marek Szykula (marek.esz@gmail.com)
 */

public class EvMessyBinaryVectorObjectiveFunctionWrapperTest 
    extends TestCase {

  // testing wrapper on underspecified individual with template
  public void testIndividualWithTemplate() {
	  
    ArrayList<Integer> genes = new ArrayList<Integer>();
    ArrayList<Boolean> alleles = new ArrayList<Boolean>();
    boolean[] template = new boolean[5];
    
    genes.add(0);alleles.add(false);
    template[0]=false;
    genes.add(1);alleles.add(true);
    template[1]=false;
    genes.add(2);alleles.add(false);
    genes.add(2);alleles.add(false);
    template[2]=false;
    template[3]=true;
    genes.add(4);alleles.add(false);
    genes.add(4);alleles.add(true);
    template[4]=true;
    
    EvMessyBinaryVectorIndividual individual =
        new EvMessyBinaryVectorIndividual(5, genes, alleles); 
    EvMessyBinaryVectorObjectiveFunctionWrapper max_sum = 
    	new EvMessyBinaryVectorObjectiveFunctionWrapper(
    	    new EvOneMax(), template);
    
    assertEquals(max_sum.evaluate(individual), 2.0);
  }

  // testing wrapper on underspecified individual using random 
  public void testIndividualWithoutTemplate() {
	  
    ArrayList<Integer> genes = new ArrayList<Integer>();
    ArrayList<Boolean> alleles = new ArrayList<Boolean>();
    
    genes.add(3);alleles.add(false);
    genes.add(4);alleles.add(false);
    genes.add(5);alleles.add(false);
    genes.add(6);alleles.add(false);

    EvMessyBinaryVectorIndividual individual =
        new EvMessyBinaryVectorIndividual(7, genes, alleles); 
    EvMessyBinaryVectorObjectiveFunctionWrapper max_sum = 
    	new EvMessyBinaryVectorObjectiveFunctionWrapper(new EvOneMax(), 3);
    
    double sum = max_sum.evaluate(individual);
    assertTrue(sum >= 1.0 && sum <= 3.0);
  }

  
  
  // testing wrapper on individual with zero genes values only
  public void testZeroIndividual() {

    ArrayList<Integer> genes = new ArrayList<Integer>();
    ArrayList<Boolean> alleles = new ArrayList<Boolean>();
    boolean[] template = new boolean[100];
    
    for (int i = 0; i < 100; i++) {
      genes.add(i);
      alleles.add(false);
      template[i]=true;
    }
    
    EvMessyBinaryVectorIndividual individual =
        new EvMessyBinaryVectorIndividual(100, genes, alleles); 
    EvMessyBinaryVectorObjectiveFunctionWrapper max_sum = 
    	new EvMessyBinaryVectorObjectiveFunctionWrapper(
    	    new EvOneMax(), template);

    assertEquals(max_sum.evaluate(individual), 0.0);
  }
  
  // testing function getPhenotype()
  public void testgetPhenotype() {
      
    ArrayList<Integer> genes = new ArrayList<Integer>();
    ArrayList<Boolean> alleles = new ArrayList<Boolean>();
    genes.add(0);alleles.add(true);
    genes.add(3);alleles.add(false);
    genes.add(4);alleles.add(true);
    genes.add(5);alleles.add(false);
    genes.add(5);alleles.add(true);
    genes.add(4);alleles.add(false);
    genes.add(6);alleles.add(false);
   
    boolean[] template = new boolean[7];
    for (int i = 0; i < 7; i++) {
      template[i]=true;
    }
    
    EvMessyBinaryVectorIndividual individual =
        new EvMessyBinaryVectorIndividual(7, genes, alleles); 
    EvMessyBinaryVectorObjectiveFunctionWrapper max_sum = 
    	new EvMessyBinaryVectorObjectiveFunctionWrapper(
    	    new EvOneMax(), template);
    
    boolean[] phenotype = max_sum.getPhenotype(individual);
    
//    assertEquals(phenotype.length, 7);
//    assertEquals(phenotype[0], true);
//    assertEquals(phenotype[1], true);
//    assertEquals(phenotype[2], true);
//    assertEquals(phenotype[3], false);
//    assertEquals(phenotype[4], true);
//    assertEquals(phenotype[5], false);
//    assertEquals(phenotype[6], false);
//    
    }
}