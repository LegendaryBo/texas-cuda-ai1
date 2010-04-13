package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation.experimental;

import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * A mutation for EvPermutationIndividuals. It randomly selects k genes in the
 * chromosome and rotates them to left. Example: Individual's chromosome is
 * {0,1,2,3,4} number of selected genes (k) is 3 Operator selected for example
 * genes at 0, 2 and 3 Then mutated individual would have chromosome {2,1,3,0,4}
 * 
 * @author Szymek Fogiel (szymek.fogiel@gmail.com)
 * @author Karol Stosiek (karol.stosiek@gmail.com)
 */
public class EvPermutationKRotationMutation extends
    EvMutation<EvPermutationIndividual> {
  private double mutation_probability;

  private int number_of_genes_mutated; // number of genes to be mutated


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
   * @param number_of_genes number of genes that participate in mutation
   */
  public EvPermutationKRotationMutation(double probability, int number_of_genes) {
    mutation_probability = probability;
    number_of_genes_mutated = number_of_genes;
  }


  @Override
  public EvPermutationIndividual mutate(EvPermutationIndividual individual) {
    if (number_of_genes_mutated > individual.getChromosome().length) {
      throw new IllegalArgumentException(
          "Can't mutate more genes then there are in the chromosome.");
    }

    /* copy of individual that will undergo mutation and be returned */
    EvPermutationIndividual mutated_individual = individual.clone();

    if (EvRandomizer.INSTANCE.nextDouble() < mutation_probability) {
      int chromosome_length = mutated_individual.getChromosome().length;

      /* we rotate genes on these positions */
      int[] positions =
          EvRandomizer.INSTANCE.nextIntList(0, chromosome_length,
              number_of_genes_mutated);

      /* get new chromosome with rotated genes */
      int[] new_chromosome =
          RotateGenes(mutated_individual.getChromosome(), positions);

      mutated_individual.setChromosome(new_chromosome); // set chromosome
    }

    return mutated_individual;
  }

}