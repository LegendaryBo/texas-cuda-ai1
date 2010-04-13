package pl.wroc.uni.ii.evolution.engine.samplealgorithms.strategies;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector.EvRealVectorExtendedStandardDeviationMutation;

/**
 * Class implements evolutionary algorithm called ES(1+1). It uses one operator:
 * ExtendedStandardDeviationMutation. It uses RealVectorIndividual as
 * individuals.
 * 
 * @author Piotr Baraniak, Tomasz Kozakiewicz
 */
public class EvOnePlusOneStrategy extends EvAlgorithm<EvRealVectorIndividual> {

  private double sigma;

  private double theta1;

  private double theta2;

  private int k;


  /**
   * Creates ES(1+1) algorithm.
   * 
   * @param sigma mutation range
   * @param k considered iteration number in the Rechenberg mutation updating
   *        strategy
   * @param theta1
   * @param theta2
   */
  public EvOnePlusOneStrategy(double sigma, int k, double theta1, double theta2) {
    super(1);

    this.sigma = sigma;
    this.k = k;
    this.theta1 = theta1;
    this.theta2 = theta2;
  }


  /**
   * Checking corectness of parameters. If all parameters are alright it
   * initialize necessary things.
   */
  @Override
  public void init() {

    EvRealVectorExtendedStandardDeviationMutation mutation_operator =
        new EvRealVectorExtendedStandardDeviationMutation(sigma, k, theta1,
            theta2);

    super.addOperatorToEnd(mutation_operator);
    super.init();

  }

}
