package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation;

import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * A mutation for EvPermutationIndividuals.
 * 
 * @author Donata Malecka, Piotr Baraniak
 */
public class EvPermutationInsertMutation extends
    EvMutation<EvPermutationIndividual> {
  private double mutation_probability;


  /**
   * Constructor.
   * 
   * @param probability probability of mutation
   */
  public EvPermutationInsertMutation(double probability) {
    mutation_probability = probability;
  }


  public EvPermutationIndividual mutate(EvPermutationIndividual individual) {

    if (EvRandomizer.INSTANCE.nextDouble() < mutation_probability) {

      int[] new_chromosome = individual.getChromosome();
      int a, b, temp;
      a = EvRandomizer.INSTANCE.nextInt(new_chromosome.length);

      while ((b = EvRandomizer.INSTANCE.nextInt(new_chromosome.length)) == a
          && new_chromosome.length > 1)
        ;

      temp = new_chromosome[a];
      if (a < b) {
        for (int j = a + 1; j <= b; j++)
          new_chromosome[j - 1] = new_chromosome[j];
        new_chromosome[b] = temp;
      } else {
        for (int j = a - 1; j >= b; j--)
          new_chromosome[j + 1] = new_chromosome[j];
        new_chromosome[b] = temp;
      }
      individual.setChromosome(new_chromosome);

    }
    return individual;
  }

}