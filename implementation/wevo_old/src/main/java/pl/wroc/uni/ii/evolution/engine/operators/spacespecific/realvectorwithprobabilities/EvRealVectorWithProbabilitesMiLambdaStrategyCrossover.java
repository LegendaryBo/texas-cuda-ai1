package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvectorwithprobabilities;

import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorWithProbabilitiesIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvCrossover;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Crossover used in ES(Mi, Lambda) and ES(Mi + Lambda). It operates on
 * RealVectorWithProbabilitiesIndividual. It takes a pair of parents from parent
 * population and randomize a number between 0 and 1 which is uses as weight.
 * Next it calculate weighted average for every part of sigma and values vectors
 * and put it into childrens. For every pair of parents there are two children.
 * 
 * @author Lukasz Witko, Piotr Baraniak
 */
public class EvRealVectorWithProbabilitesMiLambdaStrategyCrossover extends
    EvCrossover<EvRealVectorWithProbabilitiesIndividual> {

  private double random_value;


  @Override
  public int arity() {
    return 2;
  }


  @Override
  public List<EvRealVectorWithProbabilitiesIndividual> combine(
      List<EvRealVectorWithProbabilitiesIndividual> parents) {

    List<EvRealVectorWithProbabilitiesIndividual> result =
        new ArrayList<EvRealVectorWithProbabilitiesIndividual>();

    EvRealVectorWithProbabilitiesIndividual individual1, individual2;
    int dimention = parents.get(0).getDimension();

    random_value = EvRandomizer.INSTANCE.nextDouble();

    individual1 = new EvRealVectorWithProbabilitiesIndividual(dimention);
    individual1.setObjectiveFunction(parents.get(0).getObjectiveFunction());
    individual2 = new EvRealVectorWithProbabilitiesIndividual(dimention);
    individual2.setObjectiveFunction(parents.get(0).getObjectiveFunction());

    for (int j = 0; j < dimention; j++) {
      // calculate weighted value of each gene in newly created individuals
      individual1.setValue(j, parents.get(0).getValue(j) * random_value
          + (1 - random_value) * parents.get(1).getValue(j));
      individual2.setValue(j, parents.get(1).getValue(j) * random_value
          + (1 - random_value) * parents.get(0).getValue(j));
      individual1.setProbability(j, parents.get(0).getProbability(j)
          * random_value + (1 - random_value)
          * parents.get(1).getProbability(j));
      individual2.setProbability(j, parents.get(1).getProbability(j)
          * random_value + (1 - random_value)
          * parents.get(0).getProbability(j));
    }

    result.add(individual1);
    result.add(individual2);

    return result;
  }


  @Override
  public int combineResultSize() {
    return 2;
  }
}
