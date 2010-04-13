package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvOmegaIndividual;
import junit.framework.TestCase;

/**
 * Test of EvMessyAllelicDoubleMutation
 * 
 * @author Rafal Paliwoda (rp@message.pl)
 * @author Mateusz Malinowski (m4linka@gmail.com)
 */
public class EvMessyAllelicDoubleMutationTest extends TestCase {
  
  private final int genotype_length = 10;
  private final double mutation_probability = 1.0;
  private final int max_iteration = 100;
  
  public void testApplyOnOmegaIndividual() {
    boolean is_ok = true;
    for(int iteration = 0; iteration < max_iteration; iteration++) {
      EvOmegaIndividual individual = new EvOmegaIndividual(
          genotype_length,null);
      EvMessyAllelicDoubleMutation<EvOmegaIndividual> mutation =
          new EvMessyAllelicDoubleMutation<EvOmegaIndividual>(
              mutation_probability);
      EvPopulation<EvOmegaIndividual> population =
          new EvPopulation<EvOmegaIndividual>(
              new EvOmegaIndividual[] {individual});
      EvPopulation<EvOmegaIndividual> mutated_population = 
          mutation.apply(population);
      EvOmegaIndividual mutated_individual = mutated_population.get(0);
      int ind_len = individual.getChromosomeLength();
      
      for(int i = 0; i < ind_len; i++) {
        double allele1 = individual.getAllele(i);
        double allele2 = mutated_individual.getAllele(i);
        if(allele1 == allele2) {
          is_ok = false;
        }
      }
      if(!is_ok)
        break;
    }
    
    if(!is_ok) {
      fail("individual and mutated individual must be different");
    }
  }
  
}
