package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation;

import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * A mutation for EvPermutationIndividuals. Two random elements of permutation
 * will be transpose with given probability.
 * 
 * @author Donata Malecka, Piotr Baraniak
 */

public class EvPermutationTransposeMutation extends
    EvMutation<EvPermutationIndividual> {

  double mutation_probability;


  public EvPermutationTransposeMutation(double probability) {
    mutation_probability = probability;
  }


  public EvPermutationIndividual mutate(EvPermutationIndividual individual) {

    if (EvRandomizer.INSTANCE.nextDouble() < mutation_probability) {
      int a, b, temp;

      int len = individual.getChromosome().length;

      a = EvRandomizer.INSTANCE.nextInt(len);
      while ((b = EvRandomizer.INSTANCE.nextInt(len)) == a && len > 1)
        ;
      temp = individual.getGeneValue(a);
      individual.setGeneValue(a, individual.getGeneValue(b));
      individual.setGeneValue(b, temp);
    }

    return individual;
  }

}
