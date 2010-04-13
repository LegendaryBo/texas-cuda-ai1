package pl.wroc.uni.ii.evolution.engine.individuals;

import junit.framework.TestCase;
import java.util.ArrayList;

/**
 * Test of abstract EvMessyIndividual
 * on extending EvMessyBinaryVectorIndividual.
 * 
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 */

public class EvMessyBinaryVectorIndividualTest extends TestCase {
  
  public void testGetGenotype() {
    //Create messy individual
    ArrayList<Integer> genes = new ArrayList<Integer>(7);
    ArrayList<Boolean> alleles = new ArrayList<Boolean>(7);
    genes.add(0);alleles.add(true);
    genes.add(1);alleles.add(true);
    genes.add(2);alleles.add(false);
    genes.add(2);alleles.add(false);
    genes.add(2);alleles.add(true);
    genes.add(4);alleles.add(false);
    genes.add(4);alleles.add(true);
   
    EvMessyBinaryVectorIndividual individual =
      new EvMessyBinaryVectorIndividual(5, genes, alleles);    
    ArrayList<Boolean>[] genotype = individual.getGenotype();
    
    //Check alleles vector of every gene
    assertEquals(individual.getExpressedGenesNumber(), 4);
    
    assertEquals(genotype[0].size(), 1);
    assertEquals(genotype[1].size(), 1);
    assertEquals(genotype[2].size(), 3);
    assertEquals(genotype[3].size(), 0);
    assertEquals(genotype[4].size(), 2);

    assertEquals(genotype[0].get(0).booleanValue(), true);
    assertEquals(genotype[1].get(0).booleanValue(), true);
    assertEquals(genotype[2].get(0).booleanValue(), false);
    assertEquals(genotype[2].get(1).booleanValue(), false);
    assertEquals(genotype[2].get(2).booleanValue(), true);
    assertEquals(genotype[4].get(0).booleanValue(), false);
    assertEquals(genotype[4].get(1).booleanValue(), true);
    
    //Modify, get new genotype and check again
    individual.setAllele(3, 3, true);
    genotype = individual.getGenotype();
    
    assertEquals(individual.getExpressedGenesNumber(), 5);
    assertEquals(genotype[3].get(0).booleanValue(), true);
  }
  
  public void testGetCommonGenesNumber() {
    //Create 2 individuals to compare
    ArrayList<Integer> genes1 = new ArrayList<Integer>(3);
    ArrayList<Boolean> alleles1 = new ArrayList<Boolean>();
    genes1.add(1);alleles1.add(true);
    genes1.add(2);alleles1.add(true);
    genes1.add(4);alleles1.add(false);
    ArrayList<Integer> genes2 = new ArrayList<Integer>(4);
    ArrayList<Boolean> alleles2 = new ArrayList<Boolean>();
    genes2.add(0);alleles2.add(true);
    genes2.add(1);alleles2.add(true);
    genes2.add(2);alleles2.add(false);
    genes2.add(3);alleles2.add(false);
    
    EvMessyBinaryVectorIndividual individual1 =
      new EvMessyBinaryVectorIndividual(5, genes1, alleles1);    
    EvMessyBinaryVectorIndividual individual2 =
      new EvMessyBinaryVectorIndividual(5, genes2, alleles2);    
    
    //Compare the genes
    assertEquals(individual1.getCommonExpressedGenesNumber(individual2), 2);
    individual1.setAllele(2, 0, true);
    assertEquals(individual2.getCommonExpressedGenesNumber(individual1), 3);
  }
  
  public void testEquals()
  {
    //Create 3 messy individuals
    //flipped alleles and different number of expressed genes  
    ArrayList<Integer> genes1 = new ArrayList<Integer>(2);
    ArrayList<Boolean> alleles1 = new ArrayList<Boolean>();
    genes1.add(1);alleles1.add(true);
    genes1.add(2);alleles1.add(false);
    ArrayList<Integer> genes2 = new ArrayList<Integer>(2);
    ArrayList<Boolean> alleles2 = new ArrayList<Boolean>();
    genes2.add(1);alleles2.add(false);
    genes2.add(2);alleles2.add(true);
    
    EvMessyBinaryVectorIndividual individual1 =
      new EvMessyBinaryVectorIndividual(3, genes1, alleles1);    
    EvMessyBinaryVectorIndividual individual2 =
      new EvMessyBinaryVectorIndividual(3, genes2, alleles2);    
    EvMessyBinaryVectorIndividual individual3 =
      new EvMessyBinaryVectorIndividual(5, genes1, alleles1);
    
    //Modify and check
    assertTrue(!individual1.equals(individual2));
    individual1.setAllele(0, 1, false);
    assertTrue(!individual1.equals(individual2));
    individual1.setAllele(1, 2, true);
    assertTrue(individual1.equals(individual2));

    //Number of genes 3 and 5
    assertTrue(!individual1.equals(individual3));
  }
  
  @SuppressWarnings("unchecked")
  public void testClone() {
    //Create messy individual
    ArrayList<Integer> genes = new ArrayList<Integer>(2);
    ArrayList<Boolean> alleles = new ArrayList<Boolean>();
    genes.add(0);alleles.add(true);
    genes.add(1);alleles.add(false);
    
    EvMessyBinaryVectorIndividual individual =
      new EvMessyBinaryVectorIndividual(3, genes, alleles);    
    EvMessyBinaryVectorIndividual cloned_individual = individual.clone();
    
    //Check if the same 
    assertTrue(individual.equals(cloned_individual));

    //Modify and check if they are different objects
    individual.setAllele(0, 0, false);
    assertTrue(!individual.equals(cloned_individual));
  }

}