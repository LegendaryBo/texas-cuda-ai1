package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.naturalnumbervector;

import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * It modify gene's value by adding or substraction random number from given
 * interval.
 * 
 * @author Marcin Golebiowski
 */
public class EvNaturalNumberVectorChangeMutation extends
    EvMutation<EvNaturalNumberVectorIndividual> {

  private double mutation_probability;

  private int mutation_modify_max_value;


  /**
   * @param max_mutation_modify_value
   */
  public EvNaturalNumberVectorChangeMutation(double mutation_probability,
      int mutation_modify_max_value) {
    this.mutation_modify_max_value = mutation_modify_max_value;
    this.mutation_probability = mutation_probability;
  }


  public EvNaturalNumberVectorIndividual mutate(
      EvNaturalNumberVectorIndividual individual) {

    for (int position = 0; position < individual.getDimension(); position++) {
      if (EvRandomizer.INSTANCE.nextDouble() <= mutation_probability) {
        int value_mutate =
            EvRandomizer.INSTANCE.nextInt(mutation_modify_max_value + 1);
        int value_gen = individual.getNumberAtPosition(position);

        if (EvRandomizer.INSTANCE.nextBoolean()) {
          value_gen -= value_mutate;
        } else {
          value_gen += value_mutate;
        }

        if (value_gen < 0) {
          value_gen = 0;
        }
        individual.setNumberAtPosition(position, value_gen);
      }
    }
    return individual;
  }

}
