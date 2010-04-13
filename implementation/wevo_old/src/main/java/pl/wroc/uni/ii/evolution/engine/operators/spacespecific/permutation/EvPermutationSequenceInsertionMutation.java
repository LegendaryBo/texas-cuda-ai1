package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation;

import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Class implementing Sequence insertion mutation operator. This mutation
 * operator works as follows: 1) two distinct indexes i and j are picked so that
 * i < j. 2) the sequence of genes between i and j (inclusively) is cut out from
 * the chromosome. 3) another index k is picked. 4) the sequence between i and j
 * is inserted back to the chromosome, starting from position k. Example:
 * Original chromosome: (0,1,2,3,4,5,6,7,8) Index i, j, k: 5, 7, 1. Sequence:
 * (5,6,7) Resulting chromosome: (0,5,6,7,1,2,3,4,8)
 * 
 * @author Karol "Asgaroth" Stosiek (karol.stosiek@gmail.com)
 * @author Szymon Fogiel (szymek.fogiel@gmail.com)
 */
public class EvPermutationSequenceInsertionMutation extends
    EvMutation<EvPermutationIndividual> {

  private double mutation_probability;


  /**
   * Constructor for EvPermutationInversionMutation object.
   * 
   * @param probability Probability of a individual mutation.
   */
  public EvPermutationSequenceInsertionMutation(double probability) {
    this.mutation_probability = probability;
  }


  @Override
  public EvPermutationIndividual mutate(EvPermutationIndividual individual) {
    if (EvRandomizer.INSTANCE.nextDouble() >= this.mutation_probability) {
      return individual.clone();
    }

    int chromosome_length = individual.getChromosome().length;

    EvPermutationIndividual mutated_individual = individual.clone();

    int[] bounds = EvRandomizer.INSTANCE.nextIntList(0, chromosome_length, 2);

    /*
     * we set up the sequence first and last index, assuring that i < j.
     */
    int i = bounds[0];
    int j = bounds[1];
    if (bounds[0] > bounds[1]) {
      i = bounds[1];
      j = bounds[0];
    }

    /*
     * determining the maximum valid index for inserting and picking one
     */
    int max_k = chromosome_length - (j - i);
    int k = EvRandomizer.INSTANCE.nextInt(max_k);

    /*
     * manipulating the chromosome: cutting out and inserting the chosen
     * sequence
     */
    insertSequence(individual, mutated_individual, i, j, k);

    return mutated_individual;
  }


  /**
   * Cutting out and inserting back the gene sequence determined by i and j
   * indexes. Note: Indexes must hold these conditions: 1) i <= j; (non-empty
   * sequence) 2) k + j - i < chromosome length; (k index is small enough to
   * insert the sequence) 3) j < chromosome length. (the sequence must be a part
   * of the chromosome)
   * 
   * @param individual individual, whose chromosome is being modified
   * @param i index of the first gene in the sequence
   * @param j index of the last gene in the sequence
   * @param k position, where to move the sequence. After insertion, k-th gene
   *        of the chromosome is the first gene in the sequence.
   */
  private void insertSequence(EvPermutationIndividual original_individual,
      EvPermutationIndividual mutated_individual, int i, int j, int k) {

    /* we are copying the sequence */
    for (int t = 0; t < j - i + 1; t++) {
      mutated_individual.setGeneValue(k + t, original_individual.getGeneValue(i
          + t));
    }

    /* we are copying the rest */
    int t1 = 0; // iterating on original individual
    int t2 = 0; // iterating on mutated individual
    while (t1 < original_individual.getChromosome().length) {

      /*
       * if we enter the copied sequence in original individual, we jump over it
       */
      if (t1 == i) {
        t1 = j + 1;
        continue;
      }

      /*
       * if we enter the copied sequence in the mutated individual, we jump over
       * it *
       */
      if (t2 == k) {
        t2 = k + j - i + 1;
        continue;
      }

      mutated_individual.setGeneValue(t2, original_individual.getGeneValue(t1));

      t1++;
      t2++;
    }
  }
}
