package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import junit.framework.TestCase;
import java.util.ArrayList;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyBinaryVectorIndividual;

/**
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 */

public class EvMessyGenicMutationTest extends TestCase {
  
  public void testZeroApplyProbability() {
    ArrayList<Integer> genes = new ArrayList<Integer>(5);
    ArrayList<Boolean> alleles = new ArrayList<Boolean>();
    genes.add(0);alleles.add(true);
    genes.add(1);alleles.add(true);
    genes.add(2);alleles.add(false);
    
    EvMessyBinaryVectorIndividual individual =
      new EvMessyBinaryVectorIndividual(5, genes, alleles);
    
    EvMessyGenicMutation<EvMessyBinaryVectorIndividual> mutation =
      new EvMessyGenicMutation<EvMessyBinaryVectorIndividual>(0.0);
    
    EvMessyBinaryVectorIndividual mutated_individual =
        mutation.mutate(individual.clone());
    
    // Mutated individual must be unchanged
    assertEquals(individual, mutated_individual);
  }
  
  public void testOneApplyProbability() {
    ArrayList<Integer> genes = new ArrayList<Integer>(5);
    ArrayList<Boolean> alleles = new ArrayList<Boolean>();
    genes.add(10);alleles.add(true);
    genes.add(10);alleles.add(true);
    genes.add(10);alleles.add(false);
    
    EvMessyBinaryVectorIndividual individual =
      new EvMessyBinaryVectorIndividual(5, genes, alleles);
    
    EvMessyGenicMutation<EvMessyBinaryVectorIndividual> mutation =
      new EvMessyGenicMutation<EvMessyBinaryVectorIndividual>(1.0);
  
    EvMessyBinaryVectorIndividual mutated_individual =
        mutation.mutate(individual.clone());
    
    // All genes must be mutated to the range of number of expressed genes
    assertEquals(individual.getChromosomeLength(),
        mutated_individual.getChromosomeLength());
    for (int i = 0; i < mutated_individual.getChromosomeLength(); i++)
      assertTrue(mutated_individual.getGene(i) <
          mutated_individual.getGenotypeLength());
  }
  
}