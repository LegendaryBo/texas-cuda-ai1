package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector;

import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Changes each individual dimension with standard deviation. It works on
 * RealVectorIndividuals.
 * 
 * @author Piotr Baraniak, Tomasz Kozakiewicz
 */
public class EvRealVectorStandardDeviationMutation extends
    EvMutation<EvRealVectorIndividual> {

  protected double standard_deviation;


  /**
   * @param sigma mutation standard deviation
   */
  public EvRealVectorStandardDeviationMutation(double sigma) {
    standard_deviation = sigma;
  }


  public EvRealVectorIndividual mutate(EvRealVectorIndividual individual) {
    for (int j = 0; j < individual.getDimension(); j++) {
      individual.setValue(j, individual.getValue(j) + standard_deviation
          * EvRandomizer.INSTANCE.nextGaussian());
    }
    return individual;
  }

}
