package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvCrossover;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Class implementing an Order Preserving Crossover operator, presented by
 * Danuta Rutkowska, Maciej Pilinski and Leszek Rutkowski in their book "Sieci
 * neuronowe, algorytmy genetyczne i systemy rozmyte". (PWN 1997; p. 246). The
 * EvPermutationRutkowskiCrossover picks up a number of positions (dependent on
 * the fraction_unaffected parameter) in the first parent. Then, it searches for
 * the positions of the values (on the picked up positions in first parent) in
 * the second parent. Finally, it copies the genes from the first parent to the
 * second parent, omitting positions and values picked up earlier. The copying
 * operations copies values in order in which they appear in the first parent.
 * <BR>
 * For example, consider parents:<BR>
 * p = (0,1,2,3,4,5,6,7,8)<BR>
 * q = (3,0,1,7,6,5,8,2,4).<BR>
 * <BR>
 * We pick up (for example) 4 positions in p. Let it be 1,3,6 and 8. We search
 * for positions in parent q holding the same values; they are 2,4,6,8. Then, we
 * copy the rest of the values from the first parent to the second parent,
 * omitting values (1,3,6,8) and positions (2,4,6,8). It results in a child:
 * <BR>
 * o = (1,0,3,7,4,5,6,2,8).<BR>
 * 
 * @author Karol "Asgaroth" Stosiek (karol.stosiek@gmail.com)
 * @author Szymon Fogiel (szymek.fogiel@gmail.com)
 */
public class EvPermutationOPXCrossover extends
    EvCrossover<EvPermutationIndividual> {

  /**
   * Fraction of positions, that are unaffected by the crossover operator.
   */
  private double fraction_from_first_parent;


  /**
   * @param fraction_unaffected - defines the fraction of the genes, that are
   *        not affected by the crossover operator. This argument has to be in
   *        range [0,1], as it defines (by multiplying by chromosome length) the
   *        number of genes, that stay "unaffected".
   */
  public EvPermutationOPXCrossover(double fraction_from_first_parent) {
    if (fraction_from_first_parent < 0 || fraction_from_first_parent > 1)
      throw new IllegalArgumentException("fraction_from_first_parent argument"
          + " must be in range [0,1].");

    this.fraction_from_first_parent = fraction_from_first_parent;
  }


  @Override
  public int arity() {
    return 2;
  }


  @Override
  public List<EvPermutationIndividual> combine(
      List<EvPermutationIndividual> parents) {

    List<EvPermutationIndividual> result =
        new ArrayList<EvPermutationIndividual>();

    EvPermutationIndividual parent1 = parents.get(0);
    EvPermutationIndividual parent2 = parents.get(1);

    int chromosome_length = parent1.getChromosome().length;

    int positions_to_pick =
        (int) Math.floor((fraction_from_first_parent * chromosome_length));

    /*
     * Picking up positions without returning picked position back to the set.
     */
    int[] positions_to_pick_from =
        EvRandomizer.INSTANCE.nextIntList(0, chromosome_length,
            positions_to_pick);

    /*
     * creating ArrayList with the picked positions, becouse ArrayList does not
     * have a constructor from array
     */
    ArrayList<Integer> picked_positions = new ArrayList<Integer>();
    for (int i = 0; i < positions_to_pick_from.length; i++) {
      picked_positions.add(positions_to_pick_from[i]);
    }

    /*
     * creating the child chromosome, filled with and auxiliary value: -1.
     */
    int[] child_chromosome = new int[chromosome_length];
    for (int i = 0; i < chromosome_length; i++) {
      child_chromosome[i] = -1;
    }

    /* copying the values that were picked up */
    for (int i = 0; i < picked_positions.size(); i++) {
      int value = parent1.getGeneValue(picked_positions.get(i));
      child_chromosome[parent2.indexOf(value)] = value;
    }

    /*
     * copying the rest of the values from the first parent.
     */
    int j = 0; // iterate over the child chromosome
    for (int i = 0; i < chromosome_length; i++) {

      /*
       * if this is position was not copied before, we look for the first
       * position that does not contain any value, and - if we are not out of
       * bounds - we copy the value to the child chromosome.
       */
      if (!picked_positions.contains(i)) {
        while (j < child_chromosome.length && child_chromosome[j] != -1) {
          j++;
        }

        if (j < child_chromosome.length) {
          child_chromosome[j] = parent1.getGeneValue(i);
        }
      }
    }

    EvPermutationIndividual child =
        new EvPermutationIndividual(child_chromosome);

    child.setObjectiveFunction(parent1.getObjectiveFunction());

    result.add(child);
    return result;
  }


  @Override
  public int combineResultSize() {
    return 1;
  }
}
