package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.naturalnumbervector;

import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Each gene is replaced by a random number with the given mutation probability.
 * Random number is in range [0,max_value] where max value is given as a
 * parameter to constructor.
 * 
 * @author Kamil Dworakowski
 */
public class EvNaturalNumberVectorReplaceMutation extends
    EvMutation<EvNaturalNumberVectorIndividual> {

  private double mutation_probability;

  private int max_value;


  public EvNaturalNumberVectorReplaceMutation(double mutation_probability,
      int max_value) {
    this.max_value = max_value;
    this.mutation_probability = mutation_probability;
  }


  public EvNaturalNumberVectorIndividual mutate(
      EvNaturalNumberVectorIndividual individual) {

    for (int i = 0; i < individual.getDimension(); i++) {
      if (EvRandomizer.INSTANCE.nextDouble() < mutation_probability) {
        int value_mutate = EvRandomizer.INSTANCE.nextInt(max_value + 1);
        individual.setNumberAtPosition(i, value_mutate);
      }
    }
    return individual;
  }
}
