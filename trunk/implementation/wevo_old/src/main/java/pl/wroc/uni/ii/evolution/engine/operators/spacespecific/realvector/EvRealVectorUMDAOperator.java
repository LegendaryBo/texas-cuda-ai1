package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Uniform Multivariate Distribution Algorithm for continuous spaces (UMDAc).
 * 
 * @see pl.wroc.uni.ii.evolution.solutionspaces.EvRealVectorSpace
 * @author Krzysztof Sroka (krzysztof.sroka@gmail.com)
 */
public class EvRealVectorUMDAOperator implements
    EvOperator<EvRealVectorIndividual> {

  /** Size of generated populations. */
  private final int population_size;


  /**
   * Creates new UMDAc algorithm operator.
   * 
   * @param size number of individuals in generated populations
   */
  public EvRealVectorUMDAOperator(final int size) {
    if (size <= 0) {
      throw new IllegalArgumentException("Population size must be positive");
    }

    this.population_size = size;
  }


  /**
   * Applies UMDAc algorithm to sample population.
   * <p>
   * UMDAc algorithm takes a sample population (obtained earlier by some
   * selection operator) and derives mean and standard deviation parameters of
   * normal distribution, which are used to create a new population.
   * </p>
   * <p>
   * Components of individuals are independent, calculated as follows:
   * <ul>
   * <li>mean of a component's distribution is an arithmetic mean of that
   * component values in sample population</li>
   * <li>standard deviation of a component's distribution is a root mean square
   * of distances between that component values in sample population and the
   * calculated mean.
   * </ul>
   * 
   * @param population population used to derive distribution parameters
   * @return new population with values generated randomly with normal
   *         distribution
   */
  public EvPopulation<EvRealVectorIndividual> apply(
      final EvPopulation<EvRealVectorIndividual> population) {
    int individual_dimension = population.get(0).getDimension();
    double[] mean = new double[individual_dimension];
    double[] std_deviation = new double[individual_dimension];

    // calculate mean and standard deviation
    for (int i = 0; i < individual_dimension; i++) {
      for (EvRealVectorIndividual individual : population) {
        mean[i] += individual.getValue(i);
      }

      mean[i] /= population.size();

      for (EvRealVectorIndividual individual : population) {
        std_deviation[i] += Math.pow(mean[i] - individual.getValue(i), 2.0);
      }

      std_deviation[i] = Math.sqrt(std_deviation[i] / population.size());
    }

    // objective function to be applied to new individuals
    EvObjectiveFunction<?> objective_fn =
        population.get(0).getObjectiveFunction();

    // sample new population
    EvPopulation<EvRealVectorIndividual> new_population =
        new EvPopulation<EvRealVectorIndividual>(population_size);

    for (int i = 0; i < population_size; i++) {
      EvRealVectorIndividual individual =
          new EvRealVectorIndividual(individual_dimension);

      for (int j = 0; j < individual_dimension; j++) {
        individual.setValue(j, EvRandomizer.INSTANCE.nextGaussian(mean[j],
            std_deviation[j]));
      }

      individual.setObjectiveFunction(objective_fn);
      new_population.add(individual);
    }

    return new_population;
  }
}
