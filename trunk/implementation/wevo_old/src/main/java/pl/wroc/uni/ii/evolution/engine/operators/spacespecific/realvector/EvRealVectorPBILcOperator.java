package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * PBIL for continuos spaces implemented as EvOperator.
 * 
 * @author Lukasz
 * 
 */
public class EvRealVectorPBILcOperator implements
    EvOperator<EvRealVectorIndividual> {

  /**
   * Size of generated populations.
   */
  private final int population_size;

  /**
   * Learning rate.
   */
  private final double learning_rate;

  /**
   * How many best individuals are used to evaluate new deviation_vector.
   */
  private final int k_best_select;

  /**
   * Mean vector.
   */
  private double[] mean_vector;

  /**
   * Deviation vector.
   */
  private double[] deviation_vector;

  /**
   * Create new PBILc operator.
   * 
   * @param learning
   *          Learning rate
   * @param k
   *          best individuals selection
   * @param size
   *          Size of generated population
   * @param center_vector
   *          Center of space
   * @param standard_deviation_vector
   *          Standard deviation
   */
  public EvRealVectorPBILcOperator(final double learning, final int k,
      final int size, final double[] center_vector,
      final double[] standard_deviation_vector) {

    if (size <= 0) {
      throw new IllegalArgumentException("Population size must be positive");
    }

    if (center_vector == null) {
      throw new IllegalStateException(" Center of solution space is not set!");
    }

    if (standard_deviation_vector == null) {
      throw new IllegalStateException(" Deviation vector is not set!");
    }

    if (center_vector.length != standard_deviation_vector.length) {
      throw new IllegalStateException(
          " Center and deviation vectors have different dimension.");
    }

    this.population_size = size;
    this.learning_rate = learning;
    this.k_best_select = k;
    initialMeanVector(center_vector);
    initialDeviationVector(standard_deviation_vector);
  }

  /**
   * Applies PBILc algorithm to sample population.
   * 
   * @param population
   *          population used to derive distribution parameters
   * @return new population with values generated randomly with normal
   *         distribution
   */
  public EvPopulation<EvRealVectorIndividual> apply(
      final EvPopulation<EvRealVectorIndividual> population) {

    EvObjectiveFunction<EvRealVectorIndividual> objective_fn = population
        .get(0).getObjectiveFunction();

    int dimension = population.get(0).getDimension();

    if (mean_vector.length != dimension 
        || deviation_vector.length != dimension) {
      throw new IllegalArgumentException(
"Mean and deviation vectors dimension must be equal to individual dimension");
    }

    EvRealVectorIndividual[] random_population = randomPopulation(dimension,
        objective_fn);
    EvRealVectorIndividual[] best = selectKBest(random_population, 2,
        objective_fn);
    EvRealVectorIndividual worst = this.getWorstResult(random_population,
        objective_fn);

    // Iteration over bits in mean vector
    for (int k = 0; k < mean_vector.length; k++) {
      // Change mean_vector[k] by learning
      mean_vector[k] = mean_vector[k] * (1 - learning_rate)
          + (best[0].getValue(k) + best[1].getValue(k) - worst.getValue(k))
          * learning_rate;
    }

    if (k_best_select > 0) {
      EvRealVectorIndividual[] kbest;
      kbest = selectKBest(random_population, k_best_select, objective_fn);
      for (int k = 0; k < deviation_vector.length; k++) {
        deviation_vector[k] = (1 - learning_rate) * deviation_vector[k]
            + learning_rate * eval(kbest, k);
      }
    }

    // sample new population
    EvPopulation<EvRealVectorIndividual> new_population = 
      new EvPopulation<EvRealVectorIndividual>(population_size);
    
    for (int i = 0; i < population_size; i++) {
      new_population.add(random_population[i]);
    }
    return new_population;
  }

  /**
   * Generate random population.
   * @param dimension Dimension of generated individuals
   * @param objective_fn Objective function
   * @return random population
   */
  private EvRealVectorIndividual[] randomPopulation(final int dimension,
      final EvObjectiveFunction<EvRealVectorIndividual> objective_fn) {
    EvRealVectorIndividual[] pop = new EvRealVectorIndividual[population_size];

    // Adding individual to population
    for (int i = 0; i < population_size; i++) {
      pop[i] = generateIndividual();
      pop[i].setObjectiveFunction(objective_fn);
    }

    return pop;
  }

  /**
   * Gets the worst individual in current population.
   * @param source_population Source population
   * @param objective_function Objective function
   * @return the worst individual in current population
   */
  public EvRealVectorIndividual getWorstResult(
      final EvRealVectorIndividual[] source_population,
      final EvObjectiveFunction<EvRealVectorIndividual> objective_function) {
    EvRealVectorIndividual worst = source_population[0];

    for (int i = 1; i < source_population.length; i++) {
      if (objective_function.evaluate(worst) > objective_function
          .evaluate(source_population[i])) {
        worst = source_population[i];
      }
    }
    return worst;
  }

  /**
   * Initial mean vector.
   * 
   * @param center
   *          Expected values vector (center of solution space)
   */
  private void initialMeanVector(final double[] center) {
    mean_vector = new double[center.length];
    for (int i = 0; i < mean_vector.length; i++) {
      mean_vector[i] = center[i];
    }
  }

  /**
   * Initial deviation vector.
   * 
   * @param deviation
   *          Standard deviations vector
   */
  private void initialDeviationVector(final double[] deviation) {
    deviation_vector = new double[deviation.length];
    for (int i = 0; i < deviation_vector.length; i++) {
      deviation_vector[i] = deviation[i];
    }
  }

  /**
   * Generates random EvRealVectorIndividual with normal distribution.
   * 
   * @return fresh {@link EvRealVectorIndividual} with dimension equal to
   *         dimension of this space and values selected randomly (with normal
   *         distribution) from space of double values.
   */
  public EvRealVectorIndividual generateIndividual() {

    int dimension = mean_vector.length;

    EvRealVectorIndividual individual = new EvRealVectorIndividual(dimension);
    for (int i = 0; i < individual.getDimension(); i++) {
      individual.setValue(i, EvRandomizer.INSTANCE.nextGaussian(mean_vector[i],
          deviation_vector[i]));
    }

    return individual;

  }

  /**
   * Select k best individuals in source population.
   * 
   * @param source_population
   *          Source population
   * @param k
   *          How many best individuals are selected
   * @param objective_function
   *          Objective function
   * @return Array of k best individuals
   */
  private EvRealVectorIndividual[] selectKBest(
      final EvRealVectorIndividual[] source_population, final int k,
      final EvObjectiveFunction<EvRealVectorIndividual> objective_function) {
    EvRealVectorIndividual[] result = new EvRealVectorIndividual[k];
    boolean[] selected = new boolean[source_population.length];
    int selected_number;
    for (int i = 0; i < k; i++) {
      int j = 0;
      while (selected[j]) {
        j++;
      }
      EvRealVectorIndividual best = source_population[j];
      selected_number = j;
      for (; j < source_population.length; j++) {
        if (!selected[j]
            && (objective_function.evaluate(source_population[j]) 
                > objective_function.evaluate(best))) {
          best = source_population[j];
          selected_number = j;
        }
      }
      result[i] = best;
      selected[selected_number] = true;
    }

    return result;
  }

  /**
   * Used to evaluate new deviation.
   * 
   * @param best
   *          Array of best individual
   * @param gene
   *          Which gene to evaluate
   * @return new deviation
   */
  private double eval(final EvRealVectorIndividual[] best, final int gene) {
    int k = best.length;
    double average = 0;
    for (int i = 0; i < k; i++) {
      average += best[i].getValue(gene);
    }
    average /= k;
    double sum = 0;
    for (int i = 0; i < k; i++) {
      double temp = best[i].getValue(gene) - average;
      temp *= temp;
      sum += temp;
    }
    sum /= k;
    return Math.sqrt(sum);
  }

}
