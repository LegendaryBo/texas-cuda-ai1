package pl.wroc.uni.ii.evolution.objectivefunctions.messy;

import pl.wroc.uni.ii.evolution.engine.individuals.EvSimplifiedMessyIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

/**
 * Function that counts gene values of every possible Individual and return the
 * biggest sum
 * 
 * @author Krzysztof Sroka, Marcin Golebiowski, Kacper Gorski
 */

public class EvSimplifiedMessyMaxSum implements
    EvObjectiveFunction<EvSimplifiedMessyIndividual> {

  private static final long serialVersionUID = 4792892131070971818L;


  /**
   * Sumarizes individual's genes. MessyIndividual parameter must have full set
   * of genes
   * 
   * @param individual Individual to be evaluated
   */
  public double evaluate(EvSimplifiedMessyIndividual individual) {
    double suma = 0.0;

    for (int i = 0; i < individual.getLength(); i++) {
      suma += individual.getGeneValue(i);
    }
    return suma;
  }
}
