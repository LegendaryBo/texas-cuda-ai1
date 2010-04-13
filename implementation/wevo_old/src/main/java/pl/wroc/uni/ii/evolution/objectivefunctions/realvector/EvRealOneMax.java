package pl.wroc.uni.ii.evolution.objectivefunctions.realvector;

import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

/**
 * Computes sum of distances betwen dimensions
 * 
 * @author Kamil Dworakowski, Jarek Fuks
 */

public class EvRealOneMax<P extends EvRealVectorIndividual> implements
    EvObjectiveFunction<P> {

  private static final long serialVersionUID = -8046224054245242121L;


  public double evaluate(EvRealVectorIndividual individual) {
    double sum = 0;
    int dimesion = individual.getDimension();
    for (int i = 0; i < dimesion; i++) {
      sum += Math.abs(individual.getValue(i) - 1d);
    }
    return -sum;
  }

}
