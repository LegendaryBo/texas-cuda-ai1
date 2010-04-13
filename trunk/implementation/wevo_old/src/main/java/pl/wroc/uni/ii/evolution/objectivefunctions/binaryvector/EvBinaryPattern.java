package pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

/** Classic OneMax objective function for BinaryIndividual. */
public class EvBinaryPattern implements
    EvObjectiveFunction<EvBinaryVectorIndividual> {

  private static final long serialVersionUID = -6694643075129582357L;

  int[] vector;


  public EvBinaryPattern(int[] t) {
    vector = t.clone();
  }


  /**
   * Gets pattern compatibility fitness of individual. Designed & working
   * properly only for BinaryIndividual.
   */
  public double evaluate(EvBinaryVectorIndividual individual) {
    int result = 0;
    for (int i = 0; i < individual.getDimension(); i++) {
      if (individual.getGene(i) == vector[i]) {
        result += 1;
      }
    }
    return result;
  }
}
