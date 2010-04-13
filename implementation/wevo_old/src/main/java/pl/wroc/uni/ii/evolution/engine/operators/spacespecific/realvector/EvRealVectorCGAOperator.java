package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * A simple modification of CGA to real vectors. Vector of probabilities is
 * replaced with vector of means. Standard deviation is a parameter of
 * algorithm.
 * 
 * @author Marcin Brodziak
 */
public class EvRealVectorCGAOperator implements
    EvOperator<EvRealVectorIndividual> {
  private double mean[];

  private double sigma;

  private double delta;


  /**
   * Constructor
   * 
   * @param dimension -- dimension of individuals
   * @param sigma -- standard deviation
   * @param delta -- change of mean in a single iteration
   */
  public EvRealVectorCGAOperator(int dimension, double sigma, double delta) {
    mean = new double[dimension];
    this.sigma = sigma;
    this.delta = delta;
  }


  public EvPopulation<EvRealVectorIndividual> apply(
      EvPopulation<EvRealVectorIndividual> population) {
    EvRealVectorIndividual e1 = new EvRealVectorIndividual(mean.length);
    EvRealVectorIndividual e2 = new EvRealVectorIndividual(mean.length);
    for (int i = 0; i < e1.getDimension(); i++) {
      e1.setValue(i, EvRandomizer.INSTANCE.nextGaussian() * sigma + mean[i]);
      e2.setValue(i, EvRandomizer.INSTANCE.nextGaussian() * sigma + mean[i]);
    }
    e1.setObjectiveFunction(population.get(0).getObjectiveFunction());
    e2.setObjectiveFunction(population.get(0).getObjectiveFunction());

    EvRealVectorIndividual better, worse;
    if (e1.getObjectiveFunctionValue() > e2.getObjectiveFunctionValue()) {
      better = e1;
      worse = e2;
    } else {
      better = e2;
      worse = e1;
    }

    for (int i = 0; i < better.getDimension(); i++) {
      if (better.getValue(i) > worse.getValue(i)) {
        mean[i] += delta;
      } else {
        mean[i] -= delta;
      }
    }

    EvPopulation<EvRealVectorIndividual> result = population.clone();
    result.clear();
    result.add(e1);
    result.add(e2);
    return result;
  }
}
