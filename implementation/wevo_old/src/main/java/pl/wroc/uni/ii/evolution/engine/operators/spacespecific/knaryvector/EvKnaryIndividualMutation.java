package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector;

import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Simple and naive mutation for KnaryIndividuals If we mutate gene, then we
 * randomly choose value from range 0 .. individualMaxValue and set as new gene
 * value Needs serious thinking, it's just straight and simple idea
 * 
 * @author Konrad Drukala (heglion@gmail.com)
 */
public class EvKnaryIndividualMutation extends EvMutation<EvKnaryIndividual> {

  private double mutationProbability;


  /**
   * Constructor for the operator
   * 
   * @param mutationProbabilty defines the mutation probability
   */
  public EvKnaryIndividualMutation(double mutationProbability) {
    this.mutationProbability = mutationProbability;
  }


  @Override
  public EvKnaryIndividual mutate(EvKnaryIndividual individual) {
    int individual_dimension = individual.getDimension();
    int individual_max = individual.getMaxGeneValue();
    for (int i = 0; i < individual_dimension; i++) {
      if (EvRandomizer.INSTANCE.nextDouble() < mutationProbability) {
        individual.setGene(i, EvRandomizer.INSTANCE.nextInt(individual_max)); // negation
      }
    }
    return individual;
  }
}
