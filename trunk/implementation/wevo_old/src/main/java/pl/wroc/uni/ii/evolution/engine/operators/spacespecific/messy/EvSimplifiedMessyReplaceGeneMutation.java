package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import pl.wroc.uni.ii.evolution.engine.individuals.EvSimplifiedMessyIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * A mutation operator for EvMessyIndividuals. Values of every gene is replaced
 * by single random value with given probability
 * 
 * @author Marcin Golebiowski, Krzysztof Sroka
 */
public class EvSimplifiedMessyReplaceGeneMutation extends
    EvMutation<EvSimplifiedMessyIndividual> {

  private double mutation_probability;

  private int max_value_of_gene;


  /**
   * Constructor
   * 
   * @param mutation_probabilty defines the mutation probability
   * @param max_value_of_gene
   */
  public EvSimplifiedMessyReplaceGeneMutation(double mutation_probability,
      int max_value_of_gene) {
    this.mutation_probability = mutation_probability;
    this.max_value_of_gene = max_value_of_gene;
  }


  @Override
  public EvSimplifiedMessyIndividual mutate(
      EvSimplifiedMessyIndividual individual) {
    for (int position = 0; position < individual.getLength(); position++) {
      if (EvRandomizer.INSTANCE.nextDouble() <= mutation_probability) {
        individual.setGeneValue(position, EvRandomizer.INSTANCE.nextInt(0,
            max_value_of_gene, true));
      }
    }

    return individual;
  }
}
