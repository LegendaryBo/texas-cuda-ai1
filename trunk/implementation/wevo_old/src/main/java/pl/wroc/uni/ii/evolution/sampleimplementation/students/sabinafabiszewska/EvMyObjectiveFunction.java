package pl.wroc.uni.ii.evolution.sampleimplementation.students.sabinafabiszewska;

import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

/**
 * @author Sabina Fabiszewska
 */
public class EvMyObjectiveFunction implements
    EvObjectiveFunction<EvMyIndividual> {

  /**
   * 
   */
  private static final long serialVersionUID = -3224788671810971148L;


  /**
   * @param individual individual to evaluate
   * @return value of objective function of the individual
   */
  public double evaluate(final EvMyIndividual individual) {
    int sum = 0;
    for (int i = 0; i < (individual.getDimension() - 1); i++) {
      if (individual.getBit(i) != individual.getBit(i + 1)) {
        sum++;
      }
    }
    return sum;
  }

}