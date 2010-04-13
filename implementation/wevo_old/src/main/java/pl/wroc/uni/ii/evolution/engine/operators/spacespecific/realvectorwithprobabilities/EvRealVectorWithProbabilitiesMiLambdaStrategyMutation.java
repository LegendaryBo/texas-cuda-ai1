package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvectorwithprobabilities;

import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorWithProbabilitiesIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Mutation used in ES(Mi, Lambda) and ES(Mi + Lambda). It operates on
 * RealVectorWithProbabilitiesIndividual. First it changes sigmas vector by
 * multiply it for some random parameter, which depends on tau and tau_prim
 * operator parameters. Next it modifies values vector randomly with normal
 * distribution using new sigmas.
 * 
 * @author Piotr Baraniak, Tomasz Kozakiewicz
 */
public class EvRealVectorWithProbabilitiesMiLambdaStrategyMutation extends
    EvMutation<EvRealVectorWithProbabilitiesIndividual> {

  private double tau;

  private double tau_prim;


  /**
   * Modificator from tau_prim is draw once for vector, and modificator from tau
   * is draw for each part of vector
   * 
   * @param tau parameter that is used to modification probabilities vector
   * @param tau_prim parameter that is used to modification probabilities vector
   *        (sometimes it is called tau_zero)
   */
  public EvRealVectorWithProbabilitiesMiLambdaStrategyMutation(double tau,
      double tau_prim) {
    this.tau = tau;
    this.tau_prim = tau_prim;
  }


  public EvRealVectorWithProbabilitiesIndividual mutate(
      EvRealVectorWithProbabilitiesIndividual individual) {

    double epsilon0;
    double sigma;
    double x;

    epsilon0 = EvRandomizer.INSTANCE.nextGaussian() * tau_prim;

    for (int i = 0; i < individual.getDimension(); i++) {
      sigma = individual.getProbability(i);
      sigma =
          sigma
              * Math.exp(epsilon0 + EvRandomizer.INSTANCE.nextGaussian() * tau);
      individual.setProbability(i, sigma);

      x = individual.getValue(i);
      x = x + EvRandomizer.INSTANCE.nextGaussian() * sigma;
      individual.setValue(i, x);
    }
    return individual;
  }

}
