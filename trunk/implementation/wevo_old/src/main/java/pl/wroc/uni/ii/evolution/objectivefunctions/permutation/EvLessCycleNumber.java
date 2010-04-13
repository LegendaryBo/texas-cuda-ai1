package pl.wroc.uni.ii.evolution.objectivefunctions.permutation;

import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

/**
 * EvLessCycleNumber - the best is individual with one cycle
 * 
 * @author Donata Malecka, Piotr Baraniak
 */
public class EvLessCycleNumber implements
    EvObjectiveFunction<EvPermutationIndividual> {

  /**
   * 
   */
  private static final long serialVersionUID = -7533937838475216360L;


  public double evaluate(EvPermutationIndividual individual) {
    int[] chromosome = individual.getChromosome().clone();
    int cycle_number = 0;
    int k;
    for (int i = 0; i < chromosome.length; i++) {
      k = i;

      if (chromosome[k] > -1) {
        while (chromosome[k] != -1) {
          int j = k;
          k = chromosome[k];
          chromosome[j] = -1;
        }
        cycle_number++;
      }

    }
    return 1.0 / cycle_number;
  }

}
