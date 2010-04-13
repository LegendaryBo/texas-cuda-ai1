package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.milambdarokappa;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMiLambdaRoKappaIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * It is recombination (or crossover) operator. It can be used for
 * MiLambdaRoKappaIndividual. It makes recombination from 2 of given parents. It
 * choose 2 random parents. Than draw a weight of first parent (second receive 1
 * -weight). For every part of value, probability and rotation vectors it
 * calculates weighted mean from choosen parents.
 * 
 * @author Piotr Baraniak, Tomasz Kozakiewicz
 */
public class EvMiLambdaRoKappaLocalIntermediaryRecombination implements
    EvOperator<EvMiLambdaRoKappaIndividual> {

  /** First parent weight */
  private double random_u;

  /** First parent index */
  private int random_k1;

  /** Second parent index */
  private int random_k2;


  public EvPopulation<EvMiLambdaRoKappaIndividual> apply(
      EvPopulation<EvMiLambdaRoKappaIndividual> population) {

    EvPopulation<EvMiLambdaRoKappaIndividual> children =
        new EvPopulation<EvMiLambdaRoKappaIndividual>();

    EvMiLambdaRoKappaIndividual child;

    double[] values = new double[population.get(0).getDimension()];
    double[] sigma = new double[population.get(0).getDimension()];
    double[] alpha = new double[population.get(0).getAlphaLength()];

    int population_size = population.size();

    /* Drawing weight. */
    random_u = EvRandomizer.INSTANCE.nextDouble();
    /* Drawing first parent. */
    random_k1 = EvRandomizer.INSTANCE.nextInt(population_size);

    /* Drawing other parent. */
    while ((random_k2 = EvRandomizer.INSTANCE.nextInt(population_size)) == random_k1)
      ;

    EvMiLambdaRoKappaIndividual parent1 = population.get(random_k1);
    EvMiLambdaRoKappaIndividual parent2 = population.get(random_k2);

    /* Calculate means. */
    for (int i = 0; i < values.length; i++) {
      values[i] =
          random_u * parent1.getValue(i) + (1 - random_u) * parent2.getValue(i);
      sigma[i] =
          random_u * parent1.getProbability(i) + (1 - random_u)
              * parent2.getProbability(i);
    }

    for (int i = 0; i < alpha.length; i++) {
      alpha[i] =
          random_u * parent1.getAlpha(i) + (1 - random_u) * parent2.getAlpha(i);
    }
    /* Creating child. */
    child = new EvMiLambdaRoKappaIndividual(values, sigma, alpha);
    child.setObjectiveFunction(population.get(0).getObjectiveFunction());
    children.add(child);
    return children;
  }

}
