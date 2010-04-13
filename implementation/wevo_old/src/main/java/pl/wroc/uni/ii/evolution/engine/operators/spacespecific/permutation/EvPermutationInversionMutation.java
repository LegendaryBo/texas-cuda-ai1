/**
 * 
 */
package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation;

import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;
import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;

/**
 * Class implementing a well-known inversion mutation operator. Mutation that
 * takes place is simple: 1) two distinct positions are chosen; let us denote
 * them by i and j. Assume, that i < j. 2) the sequence x_{i}, x_{i+1}, ...,
 * x_{j} is inverted, i.e. we reverse the sequence and copy it to the exact same
 * place in individual.
 * 
 * @author Karol Asgaroth Stosiek (karol.stosiek@gmail.com)
 * @author Szymon Fogiel (szymek.fogiel@gmail.com)
 */
public class EvPermutationInversionMutation extends
    EvMutation<EvPermutationIndividual> {

  private double mutation_probability;


  /**
   * Constructor for EvPermutationInversionMutation object.
   * 
   * @param probability Probability of a individual mutation.
   */
  public EvPermutationInversionMutation(double probability) {
    this.mutation_probability = probability;
  }


  @Override
  public EvPermutationIndividual mutate(EvPermutationIndividual individual) {
    if (EvRandomizer.INSTANCE.nextDouble() >= this.mutation_probability) {
      return individual.clone();
    }

    EvPermutationIndividual mutated_individual = individual.clone();

    int chromosome_length = individual.getChromosome().length;

    int[] bounds = EvRandomizer.INSTANCE.nextIntList(0, chromosome_length, 2);

    /*
     * we set up the sequence first and last index, assuring that i < j.
     */
    int i = bounds[0];
    int j = bounds[1];
    if (bounds[0] > bounds[1]) {
      i = bounds[1];
      j = bounds[0];
    }

    int value_i; // gene value on position i
    int value_j; // gene value on position j
    while (i <= j) {
      value_i = individual.getGeneValue(i);
      value_j = individual.getGeneValue(j);

      /* swapping gene values */
      mutated_individual.setGeneValue(j, value_i);
      mutated_individual.setGeneValue(i, value_j);

      i++;
      j--;
    }

    return mutated_individual;
  }

}
