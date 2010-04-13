package pl.wroc.uni.ii.evolution.engine;

import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

/**
 * Simple function for testing algorithms working on 
 * EvKnaryIndividuals.<BR>
 * The function simply sums all genes' values in the
 * given individuals and return it as result.<BR>
 * Optimal solution is an individual with all bits
 * set to maximum possible value.<BR>
 * When applied to EvBinaryIndividual it works as One Max
 * function
 * 
 * 
 * @author Kacper Gorski 'admin@34all.org'
 *
 */
public class EvGenesSumFunction implements EvObjectiveFunction<EvKnaryIndividual> {

  private static final long serialVersionUID = -4290268189051886108L;

  /**
   * Sums all genes of the given individual.
   * 
   * @param individual to be evaluated
   * @return genes sum
   */
  public double evaluate(EvKnaryIndividual individual) {

    double result=0.0;
    int ind_dimension = individual.getDimension();
    
    for (int i=0; i < ind_dimension; i++) {
      result += individual.getGene(i);
    }  
      
    return result;
  }

}
