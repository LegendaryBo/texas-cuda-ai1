package pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

/**
 * Classic OneMax objective function for BinaryIndividual. It counts number of
 * bits set to "1" in individuals and returns it.<BR>
 * Optimum for this function is an obvoious string containing only true bits
 * (1,1,1,1 ...)<BR>
 * It is recomended to use it for test purposes, or to check whether
 * evolutionary algorithm works correctly.
 * 
 * @author Kacper Gorski 'admin@34all.org'
 */
public class EvOneMax implements EvObjectiveFunction<EvBinaryVectorIndividual> {

  private static final long serialVersionUID = 1845093068882807321L;


  /**
   * Counts number of true bits in given individual and returns it
   * 
   * @param individual to be evaluated.
   * @return number of true bits
   */
  public double evaluate(EvBinaryVectorIndividual individual) {
    int result = 0;
    int individual_dimension = individual.getDimension();
    for (int i = 0; i < individual_dimension; i++) {
      if (individual.getGene(i) == 1) {
        result += 1;
      }
    }
    return result;
  }
}
