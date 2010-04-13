package pl.wroc.uni.ii.evolution.objectivefunctions.omega;

import pl.wroc.uni.ii.evolution.engine.individuals.EvOmegaIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import java.util.ArrayList;

/**
 * Function that counts fixed points in the given permutation (which is
 * representing by EvOmegaIndividual).
 * 
 * @author Rafal Paliwoda (rp@message.pl)
 * @author Mateusz Malinowski (m4linka@gmail.com)
 */
public class EvOmegaMaxFixedPointsNumber implements
    EvObjectiveFunction<EvOmegaIndividual> {

  private static final long serialVersionUID = 4792131070971818928L;


  /**
   * Summarizes individual's fixed points.
   * 
   * @param individual -Individual to be evaluated
   */
  public double evaluate(EvOmegaIndividual ind) {
    double points_number = 0;
    ArrayList<Integer> fenotype = new ArrayList<Integer>(ind.getFenotype());

    for (int i = 0; i < fenotype.size(); i++) {
      if (fenotype.get(i) == i) {
        points_number++;
      }
    }
    return points_number;
  }
}
