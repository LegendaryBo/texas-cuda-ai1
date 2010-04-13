package pl.wroc.uni.ii.evolution.objectivefunctions.messy;

import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

/**
 * OneMax objective function for MessyBinaryVectorIndividual. It counts number
 * of genes set to "True" in individual.
 * 
 * @author Piotr Staszak (stachhh@gmail.com)
 * @author Marek Szykula (marek.esz@gmail.com)
 */

public class EvMessyBinaryVectorMaxSum implements
    EvObjectiveFunction<EvMessyBinaryVectorIndividual> {

  private static final long serialVersionUID = 4792892131070971818L;


  /**
   * Counts number of true genes in individual.
   * 
   * @param individual - messy individual to be evaluated
   * @return number of true genes in individual
   */
  public double evaluate(EvMessyBinaryVectorIndividual individual) {
    double suma = 0.0;

    for (int i = 0; i < individual.getChromosomeLength(); i++) {
      suma += individual.getAllele(i) ? 1.0 : 0.0;
    }
    return suma;
  }
}
