package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import junit.framework.TestCase;
import java.util.ArrayList;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy.EvMessyBinaryVectorNegationMutation;

/**
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 */

public class EvMessyBinaryVectorNegationMutationTest extends TestCase {
  
  public void testZeroApplyProbability() {
    ArrayList<Integer> genes = new ArrayList<Integer>(5);
    ArrayList<Boolean> alleles = new ArrayList<Boolean>();
    genes.add(0);alleles.add(true);
    genes.add(1);alleles.add(true);
    genes.add(2);alleles.add(false);
    
    EvMessyBinaryVectorIndividual individual =
      new EvMessyBinaryVectorIndividual(5, genes, alleles);
    
    EvMessyBinaryVectorNegationMutation mutation =
      new EvMessyBinaryVectorNegationMutation(0.0);
    
    EvMessyIndividual<Boolean> mutated_individual =
        mutation.mutate(individual.clone());

    // Mutated individual must be unchanged
    assertEquals(individual, mutated_individual);
  }
  
  public void testOneApplyProbability() {
    ArrayList<Integer> genes = new ArrayList<Integer>(5);
    ArrayList<Boolean> alleles = new ArrayList<Boolean>();
    genes.add(0);alleles.add(true);
    genes.add(1);alleles.add(true);
    genes.add(2);alleles.add(false);
    
    EvMessyBinaryVectorIndividual individual =
      new EvMessyBinaryVectorIndividual(5, genes, alleles);
    
    EvMessyBinaryVectorNegationMutation mutation =
      new EvMessyBinaryVectorNegationMutation(1.0);
    
    EvMessyIndividual<Boolean> mutated_individual =
        mutation.mutate(individual.clone());

    // Genes must be unchaned
    assertEquals(individual.getGenes(), mutated_individual.getGenes());
    // All alleles must be negated
    for (int i = 0; i < individual.getChromosomeLength(); i++)
      assertTrue(individual.getAllele(i) != mutated_individual.getAllele(i));
  }

}