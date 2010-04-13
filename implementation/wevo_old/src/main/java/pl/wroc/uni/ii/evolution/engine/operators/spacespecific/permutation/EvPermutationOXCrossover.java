package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation;

import java.util.ArrayList;
import java.util.List;
import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvCrossover;

/**
 * OX crossover. A popular crossover operator. It chooses randomly two points in
 * the chromosome (the same for both parents). Genes between these two points
 * are copied directly to the children with position preservation. The rest is
 * copied from second parent beginning from the end of the remained segment.
 * Example: Beginning of the exchange segment (which in fact is not exchanged)
 * is 3 (indexing from 0) Length of the exchange segment is 3 First parent's
 * chromosome is {1,2,3,4,5,6,7,8,9} Second parent's chromosome is
 * {4,1,2,8,7,6,9,3,5} Then children have chromosomes: First child's chromosome
 * is {2,8,7,4,5,6,9,3,1} Second child's chromosome is {3,4,5,8,7,6,9,1,2}
 * 
 * @author Szymek Fogiel (szymek.fogiel@gmail.com)
 * @author Karol Asgaroth Stosiek (karol.stosiek@gmail.com)
 */
public class EvPermutationOXCrossover extends
    EvCrossover<EvPermutationIndividual> {
  private int segment_beginning, segment_length;


  /**
   * Copying a segment to a child's chromosome from a parent's chromosome. The
   * copied segment is [from,to).
   * 
   * @param child_chromosome the child's chromosome to which segment is copied
   * @param parent the individual from which the segment is copied
   * @param from the beginning of the segment copied
   * @param to the end of the segment copied
   */
  private void CopySegment(int[] child_chromosome,
      EvPermutationIndividual parent, int from, int to) {
    if (to < from) {
      throw new IllegalArgumentException(
          "The end of the copied segment musn't be smaller then the beginning.");
    }
    if (from < 0) {
      throw new IllegalArgumentException(
          "The beginning of copied segment must be greater or equal to 0.");
    }
    if (to > child_chromosome.length) {
      throw new IllegalArgumentException(
          "The end of the copied segment must be smaller or equal"
              + " to the length of the chromosome.");
    }

    for (int i = from; i < to; i++) {

      child_chromosome[i] = parent.getGeneValue(i);
    }
  }


  /**
   * Checks if gene already exists in chromosome in segment defined by
   * segment_beginning and segment_length. If in finds a conflict it returns
   * true, else it return false.
   * 
   * @param gene the gene which we check if already exists in the segment
   * @param chromosome the chromosome in which we search for a conflict
   * @param segment_beginning the beginning of the segment in which we search
   *        for a conflict
   * @param segment_length the length of the segment in which we search for a
   *        conflict
   * @return
   */
  private boolean IsConflict(int gene, int[] chromosome, int segment_beginning,
      int segment_length) {
    int j = 0; // used for incrementation

    for (j = segment_beginning; j < segment_beginning + segment_length; j++) {

      if (gene == chromosome[j]) {
        return true;
      }
    }

    return false;
  }


  /**
   * Copies genes from the parent's to child's chromosome without affecting
   * genes in segment defined by segment beginning and segment length.
   * 
   * @param child_chromosome the chromosome to which genes are copied
   * @param parent the individual from whose chromosome genes are copied
   * @param segment_beginning the beginning of the segment which is not affected
   * @param segment_length the length of the segment which is not affected
   */
  private void ExchangeGenes(int[] child_chromosome,
      EvPermutationIndividual parent, int segment_beginning, int segment_length) {
    int chromosome_length = child_chromosome.length;
    int current_gene; // currently selected gene

    /* Filling genes for the first child */
    for (int j = segment_beginning + segment_length, i =
        segment_beginning + segment_length; j != segment_beginning; i++, j++) {

      /*
       * Checking if end of the parent chromosome, if so we return to the
       * beginning
       */
      if (i == chromosome_length) {
        i = 0;
      }

      /*
       * Checking if end of the child's chromosome, if so we return to the
       * beginning
       */
      if (j == chromosome_length) {
        j = 0;
      }

      /* Reading current gene */
      current_gene = parent.getGeneValue(i);

      /*
       * If conflict exists undo the incrementation, else write current gene to
       * child's chromosome
       */
      if (IsConflict(current_gene, child_chromosome, segment_beginning,
          segment_length)) {

        j--;
      } else {
        child_chromosome[j] = current_gene;
      }
    }
  }


  /**
   * Constructor.
   * 
   * @param segment_beginning the beginning of the exchanged segment
   * @param segment_length the length of the exchanged segment
   */
  public EvPermutationOXCrossover(int segment_beginning, int segment_length) {
    if (segment_length <= 0) {
      throw new IllegalArgumentException(
          "The length of the exchanged segment must be at least 1.");
    }
    if (segment_beginning < 0) {
      throw new IllegalArgumentException(
          "The beggining of the exchange segment must be at least 0.");
    }

    this.segment_beginning = segment_beginning;
    this.segment_length = segment_length;
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

    if (segment_beginning + segment_length > chromosome_length) {
      throw new IllegalArgumentException(
          "The end of the exchanged segment is outside the chromosome.");
    }

    int[] child1_chromosome = new int[chromosome_length];
    int[] child2_chromosome = new int[chromosome_length];

    /* Copying the exchanged segments */
    CopySegment(child1_chromosome, parent1, segment_beginning,
        segment_beginning + segment_length);
    CopySegment(child2_chromosome, parent2, segment_beginning,
        segment_beginning + segment_length);

    /* Defining the rest of the genes */
    ExchangeGenes(child1_chromosome, parent2, segment_beginning, segment_length);
    ExchangeGenes(child2_chromosome, parent1, segment_beginning, segment_length);

    EvPermutationIndividual child1 =
        new EvPermutationIndividual(child1_chromosome);

    EvPermutationIndividual child2 =
        new EvPermutationIndividual(child2_chromosome);

    child1.setObjectiveFunction(parent1.getObjectiveFunction());
    child2.setObjectiveFunction(parent2.getObjectiveFunction());

    result.add(child1);
    result.add(child2);

    return result;
  }


  @Override
  public int combineResultSize() {
    return 2;
  }
}
