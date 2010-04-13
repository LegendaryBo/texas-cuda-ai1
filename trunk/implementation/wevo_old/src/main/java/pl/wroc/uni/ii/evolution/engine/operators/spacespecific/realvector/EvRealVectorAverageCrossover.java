package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector;

import java.util.List;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvCrossover;

/**
 * Takes arity size population and returns one individual avarage on each
 * dimension
 * 
 * @author Kamil Dworakowski, Jarek Fuks
 */
public class EvRealVectorAverageCrossover extends
    EvCrossover<EvRealVectorIndividual> {

  private int arity = 2;


  public EvRealVectorAverageCrossover(int arity) {
    this.arity = arity;
  }


  @Override
  public int arity() {
    return arity;
  }


  @Override
  public List<EvRealVectorIndividual> combine(
      List<EvRealVectorIndividual> parents) {

    EvRealVectorIndividual baby =
        new EvRealVectorIndividual(parents.get(0).getDimension());
    if (arity > parents.size()) {
      throw new IllegalArgumentException("Population of size " + parents.size()
          + " cannot be used to " + " use this operator with arity " + arity
          + ".");
    }

    int dimension = baby.getDimension();

    for (int i = 0; i < arity; i++) {
      for (int dim = 0; dim < dimension; dim++) {
        baby.setValue(dim, baby.getValue(dim) + parents.get(i).getValue(dim));
      }
    }

    for (int dim = 0; dim < dimension; dim++) {
      baby.setValue(dim, baby.getValue(dim) / parents.size());
    }

    baby.setObjectiveFunction(parents.get(0).getObjectiveFunction());
    EvPopulation<EvRealVectorIndividual> results =
        new EvPopulation<EvRealVectorIndividual>(1);
    results.add(baby);
    return results;

  }


  @Override
  public int combineResultSize() {
    return 1;
  }
}
