package pl.wroc.uni.ii.evolution.sampleimplementation;

import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

/**
 * Class implementing an objective function, which calculates total track length
 * coded by the given permutation individual. In this implementation, a TSP
 * solution is encoded as follows: 1) gene value on i-th position in permutation
 * says, which city is next on the path from the city on the i-1 position. 2)
 * gene value on position 0 says, which city is next when moving from city
 * encoded by gene on the last position in the chromosome. Since wEvo framework
 * assumes, that the higher objective value individual has, the better he is,
 * the objective function returns a negative value.
 * 
 * @author Szymek Fogiel (szymek.fogiel@gmail.com)
 * @author Karol "Asgaroth" Stosiek (karol.stosiek@gmail.com)
 */

public class EvTSPSolvingObjectiveFunction implements
    EvObjectiveFunction<EvPermutationIndividual> {

  /**
   * automatically generated.
   */
  private static final long serialVersionUID = 41137291876679623L;

  private double[][] distances;


  /**
   * Constructor.
   * 
   * @param distances - matrix of distances between cities.
   */
  public EvTSPSolvingObjectiveFunction(double[][] distances) {
    this.distances = distances;
  }


  public double evaluate(EvPermutationIndividual individual) {
    double s = 0;
    int chromosome_length = individual.getChromosome().length;
    for (int i = 0; i < chromosome_length; i++) {
      s +=
          distances[individual.getGeneValue(i)][individual.getGeneValue((i + 1)
              % chromosome_length)];
    }

    return -s;
  }
}
