package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import java.util.ArrayList;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Binary Allelic Mutation negate every bit (allele in
 * MessyBinaryVectorIndividual) with given probability.
 * 
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 */

public class EvMessyBinaryVectorNegationMutation extends
    EvMutation<EvMessyBinaryVectorIndividual> {

  private double mutation_probability; // Probability of the mutation


  /**
   * Constructor creates the mutation with given probability.
   * 
   * @param mutation_probabilty defines the mutation probability
   */
  public EvMessyBinaryVectorNegationMutation(double mutation_probability) {
    if (mutation_probability < 0.0 || mutation_probability > 1.0)
      throw new IllegalArgumentException(
          "Mutation probability must be a double in range [0,1]");

    this.mutation_probability = mutation_probability;
  }


  @Override
  public EvMessyBinaryVectorIndividual mutate(
      EvMessyBinaryVectorIndividual individual) {
    if (mutation_probability == 0.0)
      return individual;

    ArrayList<Boolean> alleles = individual.getAlleles();

    // For every allele, mutate it with probability
    for (int i = 0; i < alleles.size(); i++)
      if (EvRandomizer.INSTANCE.nextDouble() < mutation_probability)
        alleles.set(i, !alleles.get(i).booleanValue());

    individual.setChromosome();
    return individual;
  }

}