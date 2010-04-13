package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation.experimental;

import java.util.ArrayList;
import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * A mutation for EvPermutationIndividuals. It randomly selects a gene's
 * position in the chromosome and finds this gene's cycle. Then it rotates genes
 * in the cycle to left. Example: Individual's chromosome is {0,1,3,4,2}
 * Operator selected for example gene at position 2 (that is 3) Then the
 * selected cycle would be (3,4,2) and mutated individual would have chromosome
 * {0,1,4,2,3}
 * 
 * @author Szymek Fogiel (szymek.fogiel@gmail.com)
 * @author Karol Stosiek (karol.stosiek@gmail.com)
 */
public class EvPermutationCycleRotationMutation extends
    EvMutation<EvPermutationIndividual> {
  private double mutation_probability;


  /**
   * Find a cycle in a chromosome starting on position pointed by position
   * 
   * @param position position on which the cycle starts
   * @param chromosome chromosome in which we search for cycle
   * @return table of positions of genes that are in the found cycle
   */
  private int[] FindCycle(int[] chromosome, int position) {
    if (position >= chromosome.length) {
      throw new IllegalArgumentException("Position exceedes chromosome.");
    }
    /* holds positions of the cycle */
    ArrayList<Integer> cycle_list = new ArrayList<Integer>();
    int current_position = position; // points currently selected position
    cycle_list.add(current_position);
    current_position = chromosome[current_position];

    /* finding cycle */
    while (current_position != position) {
      cycle_list.add(current_position);
      current_position = chromosome[current_position];
    }

    /* table with cycle's positions to return */
    int[] cycle_positions = new int[cycle_list.size()];

    /* coping cycle_list to cycle_positions */
    for (int i = 0; i < cycle_positions.length; i++) {
      cycle_positions[i] = cycle_list.get(i);
    }

    return cycle_positions;
  }


  /**
   * Rotates genes in chromosome on given positions.
   * 
   * @param chromosome chromosome in which genes are rotated
   * @param positions positions on which genes are rotated
   * @return chromosome with rotated genes
   */
  private int[] RotateGenes(int[] chromosome, int[] positions) {
    int temp_gene; // needed to memorize one gene

    /* Rotating genes */
    temp_gene = chromosome[positions[0]]; // memorize first gene
    for (int i = 0; i < positions.length - 1; i++) {
      chromosome[positions[i]] = chromosome[positions[i + 1]];
    }
    chromosome[positions[positions.length - 1]] = temp_gene;

    return chromosome;
  }


  /**
   * Constructor.
   * 
   * @param probability probability of mutation
   */
  public EvPermutationCycleRotationMutation(double probability) {
    mutation_probability = probability;
  }


  @Override
  public EvPermutationIndividual mutate(EvPermutationIndividual individual) {
    /* copy of individual that will undergo mutation and be returned */
    EvPermutationIndividual mutated_individual = individual.clone();

    if (EvRandomizer.INSTANCE.nextDouble() < mutation_probability) {
      int chromosome_length = mutated_individual.getChromosome().length;

      /* this will be the first gene in the cycle to rotate */
      int selected_gene_position =
          EvRandomizer.INSTANCE.nextInt(0, chromosome_length);

      /* we rotate genes on these positions */
      int[] positions =
          FindCycle(mutated_individual.getChromosome(), selected_gene_position);

      /* get new chromosome with rotated genes */
      int[] new_chromosome =
          RotateGenes(mutated_individual.getChromosome(), positions);

      mutated_individual.setChromosome(new_chromosome); // set chromosome
    }

    return mutated_individual;
  }

}