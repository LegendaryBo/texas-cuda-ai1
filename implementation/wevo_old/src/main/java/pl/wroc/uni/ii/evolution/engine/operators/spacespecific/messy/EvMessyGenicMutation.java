package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import java.util.ArrayList;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Genic Mutation with given probability change randomly (uniformly) gene for
 * every allele in chromosome.
 * 
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 * @param <T> type of EvMessyIndividual
 */

public class EvMessyGenicMutation<T extends EvMessyIndividual> extends
    EvMutation<T> {

  private double mutation_probability; // Probability of the mutation


  /**
   * Constructor creates the mutation with given probability.
   * 
   * @param mutation_probabilty defines the mutation probability
   */
  public EvMessyGenicMutation(double mutation_probability) {
    if (mutation_probability < 0.0 || mutation_probability > 1.0)
      throw new IllegalArgumentException(
          "Mutation probability must be a double in range [0,1]");

    this.mutation_probability = mutation_probability;
  }


  @Override
  @SuppressWarnings("unchecked")
  public T mutate(T individual) {
    if (mutation_probability == 0.0)
      return individual;

    ArrayList<Integer> genes = individual.getGenes();
    int number_of_genes = individual.getGenotypeLength();

    // Mutate every gene with given probability
    for (int i = 0; i < genes.size(); i++)
      if (EvRandomizer.INSTANCE.nextDouble() < mutation_probability)
        genes.set(i, EvRandomizer.INSTANCE.nextInt(number_of_genes));

    individual.setChromosome();
    return individual;
  }

}