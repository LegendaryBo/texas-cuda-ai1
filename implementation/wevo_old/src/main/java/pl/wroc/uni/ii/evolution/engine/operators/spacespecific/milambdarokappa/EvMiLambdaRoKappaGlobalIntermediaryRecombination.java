package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.milambdarokappa;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMiLambdaRoKappaIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * It is recombination (or crossover) operator. It can be used for
 * MiLambdaRoKappaIndividual. It makes recombination from all given parents. For
 * every part of value, probability and rotation vectors it calculates
 * arithmetic mean from all parents.
 * 
 * @author Piotr Baraniak, Tomasz Kozakiewicz
 */
public class EvMiLambdaRoKappaGlobalIntermediaryRecombination implements
    EvOperator<EvMiLambdaRoKappaIndividual> {

  public EvPopulation<EvMiLambdaRoKappaIndividual> apply(
      EvPopulation<EvMiLambdaRoKappaIndividual> population) {
    EvPopulation<EvMiLambdaRoKappaIndividual> children =
        new EvPopulation<EvMiLambdaRoKappaIndividual>();
    EvMiLambdaRoKappaIndividual child;
    double[] values = new double[population.get(0).getDimension()];
    double[] sigma = new double[population.get(0).getDimension()];
    double[] alpha = new double[population.get(0).getAlphaLength()];
    int population_size = population.size();
    /* Calculating mean of values and probabilities from parents. */
    for (int i = 0; i < values.length; i++) {
      values[i] = 0;
      sigma[i] = 0;
      for (int j = 0; j < population_size; j++) {
        values[i] += population.get(j).getValue(i);
        sigma[i] += population.get(j).getProbability(i);
      }
      values[i] /= population_size;
      sigma[i] /= population_size;
    }
    /* Calculating mean of rotation parameters from parents. */
    for (int i = 0; i < alpha.length; i++) {
      alpha[i] = 0;
      for (int j = 0; j < population_size; j++) {
        alpha[i] += population.get(j).getAlpha(i);
      }
      alpha[i] /= population_size;
    }
    /* Creating one child. */
    child = new EvMiLambdaRoKappaIndividual(values, sigma, alpha);
    child.setObjectiveFunction(population.get(0).getObjectiveFunction());
    children.add(child);
    return children;
  }

}
