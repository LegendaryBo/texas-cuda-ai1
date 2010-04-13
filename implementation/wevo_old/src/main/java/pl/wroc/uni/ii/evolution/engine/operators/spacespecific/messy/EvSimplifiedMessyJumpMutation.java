package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import java.util.ArrayList;
import pl.wroc.uni.ii.evolution.engine.individuals.EvSimplifiedMessyIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * For every gene move randomly choosen gene's values to another gene with given
 * probability.
 * 
 * @author Marcin Golebiowski
 */
public class EvSimplifiedMessyJumpMutation extends
    EvMutation<EvSimplifiedMessyIndividual> {

  private double mutation_probablity;


  /**
   * Constructor
   * 
   * @param mutation_probabilty defines the mutation probability
   */
  public EvSimplifiedMessyJumpMutation(double mutation_probablity) {
    this.mutation_probablity = mutation_probablity;
  }


  @Override
  public EvSimplifiedMessyIndividual mutate(
      EvSimplifiedMessyIndividual individual) {

    /** iterate over genes * */
    for (int position = 0; position < individual.getLength(); position++) {

      /** check if do mutation * */
      if (individual.getGeneValues(position).size() != 0
          && EvRandomizer.INSTANCE.nextDouble() <= mutation_probablity) {

        /** choose target gene * */
        int target = EvRandomizer.INSTANCE.nextInt(individual.getLength());

        /** check if target is different than current gene * */
        if (target != position) {

          /** decide how many gene's values will be moved * */
          int how_many =
              EvRandomizer.INSTANCE.nextInt(0, individual.getGeneValues(
                  position).size(), true);

          if (how_many != 0) {

            boolean[] which =
                EvRandomizer.INSTANCE.nextBooleanList(individual.getGeneValues(
                    position).size(), how_many);

            ArrayList<Integer> from_values_to_moved = new ArrayList<Integer>();
            ArrayList<Integer> from_values_to_stay = new ArrayList<Integer>();

            /** select gene values to stay and to moved * */
            for (int j = 0; j < which.length; j++) {
              if (which[j]) {
                from_values_to_moved.add(individual.getGeneValue(position, j));
              } else {
                from_values_to_stay.add(individual.getGeneValue(position, j));
              }
            }

            individual.setGeneValues(position, from_values_to_stay);
            individual.addGeneValues(target, from_values_to_moved);
            individual.removeDuplicateGeneValues(target);

          }

        }
      }

    }
    return individual;

  }
}
