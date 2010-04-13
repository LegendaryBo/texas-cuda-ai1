package pl.wroc.uni.ii.evolution.objectivefunctions.naturalnumbervector;

import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

/**
 * Classic Pattern objective function for EvNaturalPatternIndividuals.
 * 
 * @author Marcin Golebiowski
 */
public class EvNaturalPattern implements
    EvObjectiveFunction<EvNaturalNumberVectorIndividual> {

  private static final long serialVersionUID = -7816623501435772332L;

  int[] vector;


  /**
   * Sets a given pattern.
   * 
   * @param pattern
   */
  public EvNaturalPattern(int[] pattern) {
    vector = pattern.clone();
  }


  /**
   * Gets onemax fitness of individual. Designed & working properly only for
   * BinaryIndividual.
   */
  public double evaluate(EvNaturalNumberVectorIndividual individual) {
    if (individual.getDimension() != vector.length) {
      throw new IllegalArgumentException(
          "Dimension of NaturalNumberIndividual must "
              + "be consistent with length of the pattern");
    }

    double result = 0.0;
    for (int i = 0; i < vector.length; i++) {
      if (vector[i] == individual.getNumberAtPosition(i))
        result += 1.0;
    }
    return result;
  }
}
