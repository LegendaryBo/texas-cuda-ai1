/**
 * 
 */
package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation.experimental;

import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Class implementing so-called shuffling mutation. The mutation process
 * consists of: 2) Picking up k positions from the chromosome; 3) Shuffling
 * values between these positions. Important: k value depends on the fraction
 * parameter; it means that k = floor(chromosome_length * fraction).
 * 
 * @author Karol Asgaroth Stosiek (karol.stosiek@gmail.com)
 * @author Szymon Fogiel (szymek.fogiel@gmail.com)
 */
public class EvPermutationShuffleMutation extends
    EvMutation<EvPermutationIndividual> {

  double mutation_probability;

  double fraction;


  /**
   * Constructor of EvPermutationShuffleMutation.
   * 
   * @param probability probability of mutation
   * @param fraction fraction of genes that are going to be shuffled
   */
  public EvPermutationShuffleMutation(double probability, double fraction) {
    this.mutation_probability = probability;
    this.fraction = fraction;
  }


  @Override
  public EvPermutationIndividual mutate(EvPermutationIndividual individual) {
    if (EvRandomizer.INSTANCE.nextDouble() >= this.mutation_probability) {
      return individual.clone();
    }

    EvPermutationIndividual mutated_individual = individual.clone();

    int chromosome_length = individual.getChromosome().length;
    int k = (int) Math.floor(this.fraction * chromosome_length);

    /* picking up positions to shuffle */
    int[] positions =
        EvRandomizer.INSTANCE.nextIntList(0, chromosome_length, k);

    /*
     * a map of positions to substitute with. i-th value in the original
     * individual is substituted with value on position
     * position[positions_permutation[i]]
     */
    int[] positions_permutation =
        EvRandomizer.INSTANCE.nextPermutation(positions.length);

    int value; // value to substitute the original
    for (int i = 0; i < positions.length; i++) {
      value = individual.getGeneValue(positions[positions_permutation[i]]);

      mutated_individual.setGeneValue(positions[i], value);
    }

    return mutated_individual;
  }

}
