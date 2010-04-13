package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.permutation;

import java.util.ArrayList;
import java.util.List;
import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvCrossover;

/**
 * Cycle crossover (CX) for EvPermutationIndividuals. Cycle crossover operator
 * was first introduced by Oliver, Smith and Holland in "A Study of Permutation
 * Crossover Operators on the Traveling Salesman Problem" (1987) It crosses the
 * cycles between parent individuals, resulting in two children, which have
 * cycles from both first and second parent. In details, the CX operator: 1)
 * identifies the so-called cycles in both parents; a cycle is created in a
 * following manner:<BR>
 * a. A gene with lowest index (among unused indexes) is chosen in a parent; let
 * us denote it by i. In a first step of the cycles identification process, it
 * would be 1. <BR>
 * b. Let v1 and v2 be the gene values on i-th position in first and second
 * parent, respectively. We search for the gene with value v2 in the first
 * parent; let i' denote its position. We add v2 to the cycle. (we don't add v2
 * physically to the cycle; from now on, we only consider v2 to be in that
 * cycle). <BR>
 * c. We repeat step b with i = i' until we close the cycle.<BR>
 * d. We repeat steps a - c until all cycles are identified. <BR>
 * 2) the first child is created from cycles copied from alternating parents,
 * starting from first parent. The second child is created from cycles copied
 * from alternating parents, starting from the second parent.<BR>
 * Note: It is important, that copying cycles preserves the original positions
 * of genes in their parents.<BR>
 * Example:<BR>
 * Parent 1: 8 4 7 3 6 2 5 1 9 0<BR>
 * Parent 2: 0 1 2 3 4 5 6 7 8 9<BR>
 * Identified cycles: (8 0 9)<BR>
 * (4 1 7 2 5 6)<BR>
 * (3)<BR>
 * Result:<BR>
 * Child 1: 8 1 2 3 4 5 6 7 9 0<BR>
 * Child 2: 0 4 7 3 6 2 5 1 8 9 <BR>
 * 
 * @author Karol "Asgaroth" Stosiek (karol.stosiek@gmail.com)
 * @author Szymek Fogiel (szymek.fogiel@gmail.com)
 */
public class EvPermutationCXCrossover extends
    EvCrossover<EvPermutationIndividual> {

  @Override
  public int arity() {
    return 2;
  }


  @Override
  public List<EvPermutationIndividual> combine(
      List<EvPermutationIndividual> parents) {

    List<EvPermutationIndividual> result =
        new ArrayList<EvPermutationIndividual>();

    int[] chromosome_parent1 = parents.get(0).getChromosome();
    int[] chromosome_parent2 = parents.get(1).getChromosome();

    int chromosome_length = chromosome_parent1.length;

    int[] chromosome_child1 = new int[chromosome_length];
    int[] chromosome_child2 = new int[chromosome_length];

    /*
     * clearing the child chromosomes; it is important for the cycle
     * identification to work.
     */
    for (int i = 0; i < chromosome_length; i++) {
      chromosome_child1[i] = -1;
      chromosome_child2[i] = -1;
    }

    int[] values_to_positions_parent1 = new int[chromosome_length];

    /*
     * mapping gene values to their positions in parental chromosomes.
     */
    for (int i = 0; i < chromosome_length; i++) {
      values_to_positions_parent1[chromosome_parent1[i]] = i;
    }

    /*
     * Crossing over; identifying cycles and copying them to the children.
     */
    int j = 0; // iterating over a single cycle
    int v1 = 0; // j-th gene value of the first parent
    int v2 = 0; // j-th gene value of the second parent
    int cycles = 0; // number of cycles identified yet

    for (int i = 0; i < chromosome_length; i++) {

      /*
       * if i-th gene was not considered yet, start iterating over a cycle
       * beginning with that gene and copy the cycle values to appropriate
       * children.
       */
      if (chromosome_child1[i] == -1) {
        j = i;
        do {
          v1 = chromosome_parent1[j];
          v2 = chromosome_parent2[j];

          /* This condition "switches" the children. */
          if (cycles % 2 == 0) {
            chromosome_child1[j] = v1;
            chromosome_child2[j] = v2;
          } else {
            chromosome_child1[j] = v2;
            chromosome_child2[j] = v1;
          }

          j = values_to_positions_parent1[v2];
        } while (i != j);

        cycles++;
      }
    }

    EvPermutationIndividual child1 =
        new EvPermutationIndividual(chromosome_child1);

    EvPermutationIndividual child2 =
        new EvPermutationIndividual(chromosome_child2);

    child1.setObjectiveFunction(parents.get(0).getObjectiveFunction());
    child2.setObjectiveFunction(parents.get(1).getObjectiveFunction());

    result.add(child1);
    result.add(child2);

    return result;
  }


  @Override
  public int combineResultSize() {
    return 2;
  }
}
