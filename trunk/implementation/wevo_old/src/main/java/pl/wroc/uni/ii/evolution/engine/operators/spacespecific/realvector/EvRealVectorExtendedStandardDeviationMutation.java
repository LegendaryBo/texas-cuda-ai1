package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector;

import java.util.LinkedList;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;

/**
 * StandardDeviationMutation extended to include Rechenberg's rule to change
 * sigma (standard deviation) parameter. It works on RealVectorIndividuals.
 * 
 * @author Piotr Baraniak, Tomasz Kozakiewicz
 */
public class EvRealVectorExtendedStandardDeviationMutation extends
    EvRealVectorStandardDeviationMutation {

  private double theta1;

  private double theta2;

  private int iteration_number;

  private LinkedList<Boolean> success_history;

  private int success_iterations;


  /**
   * @param sigma mutation standard deviation
   * @param iteration_number number of iteration for Rechenberg's rule
   * @param theta1 multiplier for more than 20% of successes
   * @param theta2 multiplier for less than 20% of successes
   */
  public EvRealVectorExtendedStandardDeviationMutation(double sigma,
      int iteration_number, double theta1, double theta2) {
    super(sigma);
    this.theta1 = theta1;
    this.theta2 = theta2;
    this.iteration_number = iteration_number;
    success_iterations = 0;
    success_history = new LinkedList<Boolean>();
  }


  public EvRealVectorIndividual mutate(EvRealVectorIndividual individual) {
    super.setMutateClone(true);

    EvRealVectorIndividual individual2 = super.mutate(individual);
    EvRealVectorIndividual best = individual;

    if (individual.getObjectiveFunctionValue() < individual2
        .getObjectiveFunctionValue()) {
      success_iterations++;
      success_history.add(true);
      best = individual2;
    } else {
      success_history.add(false);
    }

    if (success_history.size() >= iteration_number) {
      rechenbergChange();
      boolean temp = (Boolean) success_history.getFirst();
      success_history.removeFirst();
      if (temp) {
        success_iterations--;
      }
    }
    return best;
  }


  private void rechenbergChange() {
    // System.out.println(standard_deviation);
    if (1.0 * success_iterations / iteration_number > 0.2) {
      // System.out.println("sukces");
      standard_deviation = standard_deviation * theta1;
      return;
    }
    if (1.0 * success_iterations / iteration_number < 0.2) {
      // System.out.println("porazka");
      standard_deviation = standard_deviation * theta2;
      return;
    }
  }
}
