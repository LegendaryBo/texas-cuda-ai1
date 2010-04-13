package pl.wroc.uni.ii.evolution.sampleimplementation.students.sabinafabiszewska;

import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * @author Sabina Fabiszewska
 */
public class EvMyMutation extends EvMutation<EvMyIndividual> {

  /**
   * 
   */
  private final double probability;


  /**
   * @param probab probability of mutation of each gene
   */
  EvMyMutation(final double probab) {
    this.probability = probab;
  }


  /**
   * @param individual which will be mutated
   * @return mutated individual
   */
  @Override
  public EvMyIndividual mutate(final EvMyIndividual individual) {
    for (int i = 0; i < individual.getDimension(); i++) {
      if (EvRandomizer.INSTANCE.nextProbableBoolean(probability)) {
        individual.setBit(i, !individual.getBit(i));
      }
    }
    return individual;
  }
}