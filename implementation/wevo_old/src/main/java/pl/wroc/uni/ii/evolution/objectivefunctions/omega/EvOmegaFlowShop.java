package pl.wroc.uni.ii.evolution.objectivefunctions.omega;

import java.util.ArrayList;

import pl.wroc.uni.ii.evolution.engine.individuals.EvOmegaIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

/**
 * Function that solves FlowShop problem.
 * 
 * @author Rafal Paliwoda (rp@message.pl)
 * @author Mateusz Malinowski (m4linka@gmail.com)
 */
public class EvOmegaFlowShop implements EvObjectiveFunction<EvOmegaIndividual> {

  private static final long serialVersionUID = -129872498273492347L;

  private final int number_of_machines;

  private final int number_of_jobs;

  private final double time[][];


  public EvOmegaFlowShop(double time[][]) {

    this.number_of_jobs = time[0].length;
    this.number_of_machines = time.length;
    this.time = time;
  }


  public double evaluate(EvOmegaIndividual individual) {
    double min_time[] = new double[number_of_jobs];
    for (int i = 0; i < number_of_jobs; i++) {
      min_time[i] = 0;
    }

    ArrayList<Integer> solution = individual.getFenotype();
    for (int i = 0; i < number_of_machines; i++) {
      min_time[0] += time[i][solution.get(0)];
      for (int j = 1; j < number_of_jobs; j++) {
        min_time[j] =
            Math.max(min_time[j], min_time[j - 1]) + time[i][solution.get(j)];
      }
    }
    return -min_time[number_of_jobs - 1];
  }

}
