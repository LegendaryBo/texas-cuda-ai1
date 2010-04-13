package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * A standard mutation for EvBinaryVectorIndividual. Value of every gene is
 * negated with given probability.
 * 
 * @author: Piotr Baraniak, Marek Chrusciel, Marcin Golebiowski
 */
public class EvBinaryVectorNegationMutation extends
    EvMutation<EvBinaryVectorIndividual> {
  private double mutation_probability;


  /**
   * Constructor for the operator
   * 
   * @param mutation_probabilty defines the mutation probability
   */
  public EvBinaryVectorNegationMutation(double mutation_probabilty) {
    this.mutation_probability = mutation_probabilty;
  }


  @Override
  public EvBinaryVectorIndividual mutate(EvBinaryVectorIndividual individual) {

    int individual_dimension = individual.getDimension();

    for (int i = 0; i < individual_dimension; i++) {
      if (EvRandomizer.INSTANCE.nextDouble() < mutation_probability) {
        individual.setGene(i, (individual.getGene(i) + 1) % 2); // negation
      }
    }
    return individual;
  }
}
