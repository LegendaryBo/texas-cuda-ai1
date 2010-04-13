package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.bmda;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;

/**
 * Compute Pearson's chi-square statistic for binary individuals.
 * 
 * The test is definied by: X^2 = sum( (observed - expected)^2 / expected)
 * 
 * So X^2(i,j) = sum( (N*p(xi,xj) - N*p(xi)*p(xj))^2 / N*p(xi)*p(xj) )
 * 
 * Where i and j are two variables to check and i != j N is the size of
 * population p(x) is probability to get value x p(x1,x2) is probability to get
 * value x1 and x2 sum is sum on the every combination of value x1 and x2
 * 
 * If X^2 < 3.84 then variables are independent in 95%.
 * 
 * @author Mateusz Poslednik mateusz.poslednik@gmail.com
 * 
 */
public class EvPearsonsChiSquareStatistic {

  /** Population. */
  private EvPopulation<EvBinaryVectorIndividual> population;

  /**
   * Constructor.
   * 
   * @param population_
   *          population
   */
  public EvPearsonsChiSquareStatistic(
      final EvPopulation<EvBinaryVectorIndividual> population_) {
    this.population = population_;
  }

  /**
   * Compute Pearson's chi-square statistic. X^2(i,j) = sum( (N*p(xi,xj) -
   * N*p(xi)*p(xj))^2 / N*p(xi)*p(xj) ) where i != j i and j < length of
   * chromosome i and j >= 0
   * 
   * @param i
   *          number of i-th gene
   * @param j
   *          number of j-th gene
   * @return X^2(i,j) - Pearson's chi-square
   */
  public double computeX(final int i, final int j) {
    double x = 0;
    int dimension = population.get(0).getDimension();
    int populationSize = population.size();
    // probability to get any combination or value:
    double pi1 = 0; // probability to get 1 by i-th gene
    double pj1 = 0; // end so on...
    double pij00 = 0;
    double pij01 = 0;
    double pij10 = 0;
    double pij11 = 0;
    // control arguments
    if (i < 0 || j < 0) {
      throw new IllegalArgumentException(
          "Parameters i and j must be higher than 0.");
    }
    if (i >= dimension || j >= dimension) {
      throw new IllegalArgumentException(
          "Parameters i and j must be less than lenght of chromosone.");
    }
    if (i == j) {
      throw new IllegalArgumentException(
          "Parameters i and j must be different.");
    }
    // start algorithm
    // compute amount of occurrence of each value the two variables.
    for (int index = 0; index < populationSize; index++) {
      if (population.get(index).getGene(i) == 1) {
        pi1++;
        if (population.get(index).getGene(j) == 1) {
          pj1++;
          pij11++;
        } else {
          pij10++;
        }
      } else {
        if (population.get(index).getGene(j) == 1) {
          pj1++;
          pij01++;
        } else {
          pij00++;
        }
      }
    }
    // compute probability
    pi1 /= populationSize;
    pj1 /= populationSize;
    pij00 /= populationSize;
    pij01 /= populationSize;
    pij10 /= populationSize;
    pij11 /= populationSize;
    // compute X^2(i,j)
    x += computePart(populationSize, 1.0 - pi1, 1.0 - pj1, pij00);
    x += computePart(populationSize, 1.0 - pi1, pj1, pij01);
    x += computePart(populationSize, pi1, 1.0 - pj1, pij10);
    x += computePart(populationSize, pi1, pj1, pij11);
    return x;
  }

  /**
   * Compute internal part of X^2:. (N*p(xi,xj) - N*p(xi)*p(xj))^2 /
   * N*p(xi)*p(xj)
   * 
   * @param n
   *          population size
   * @param p1
   *          probability to get the value on first gen
   * @param p2
   *          probability to get the value on second gen
   * @param p1p2
   *          probability to get the value on the both gens
   * @return part of X^2
   */
  private double computePart(final int n, final double p1, final double p2,
      final double p1p2) {
    if (p1 == 0.0 || p2 == 0.0) {
      return 0.0;
    }
    double up = (n * p1p2) - (n * p1 * p2);
    up *= up;
    double down = n * p1 * p2;
    return up / down;
  }

  /**
   * Getter.
   * 
   * @return population
   */
  public EvPopulation<EvBinaryVectorIndividual> getPopulation() {
    return population;
  }

  /**
   * Setter.
   * 
   * @param population_
   *          New population.
   */
  public void setPopulation(
      final EvPopulation<EvBinaryVectorIndividual> population_) {
    this.population = population_;
  }
}
