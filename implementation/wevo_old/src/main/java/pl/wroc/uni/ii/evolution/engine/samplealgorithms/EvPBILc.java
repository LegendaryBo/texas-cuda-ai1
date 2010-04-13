package pl.wroc.uni.ii.evolution.engine.samplealgorithms;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.solutionspaces.EvRealVectorSpace;

/**
 * Implementation of PBILc . 
 * 
 * @author Lukasz Wojtus
 */
public final class EvPBILc extends EvAlgorithm<EvRealVectorIndividual>  {

  /**
   * Solution space for which CGA works.
   */
  private EvRealVectorSpace solution_space;

  /**
   * Learning rate.
   */
  private final double learning_rate;

  /**
   * Mutation probability.
   */
  private final double mutation_probability;

  /**
   * Mutation rate.
   */
  private final double mutation_rate;

  /**
   * Mean vector.
   */
  private double[] mean_vector;

  /**
   * Deviation vector.
   */
  private double[] deviation_vector;
  
  /**
   * Some population for learning probablity vector.
   */
  private EvRealVectorIndividual[] population;
  
  /**
   * 
   * @param theta_1 learning rate
   * @param theta_2 mutation probability
   * @param theta_3 mutation rate
   * @param population_size population size
   * @param center_vector center of search space
   * @param standard_deviation_vector standard deviation vector 
   */
  public EvPBILc(final double theta_1, final double theta_2,
      final double theta_3, final int population_size,
      final double[] center_vector,
      final double[] standard_deviation_vector) {
    super(population_size);
    this.learning_rate = theta_1;
    this.mutation_probability = theta_2;
    this.mutation_rate = theta_3;
    initialMeanVector(center_vector);
    initialDeviationVector(standard_deviation_vector);
  }
  
  /**
   * Check given data.
   */
  public void init() {
    if (solution_space == null) {
      throw new IllegalStateException("Solution space is not set!");
    }

    if (mean_vector == null)  {
      throw new IllegalStateException(" Center of solution space is not set!");
    }

    if (deviation_vector == null)  {
      throw new IllegalStateException(" Deviation vector is not set!");
    }
    
    if (mean_vector.length != deviation_vector.length)  {
      throw new IllegalStateException(
          " Center and deviation vectors have different dimension.");
    }
    
    if (mean_vector.length != solution_space.getDimension())  {
      throw new IllegalStateException(
  " Center and deviation vectors dimension must be equal to space dimension.");
    }
  }
  
  /**
   * Sets solution space for PBILc.
   * @param space Solution space.
   */
  public void setSolutionSpace(
          final EvSolutionSpace<EvRealVectorIndividual> space) {
    solution_space = (EvRealVectorSpace) space;
  }

  /**
   * A single iteration for PBILc. Based on "Extending Population-Based 
   * Incremental Learning to Continuous Search Spaces" by Michele Sebag and 
   * Antoine Ducoulombier.
   *
   */
  @Override
  public void doIteration() {

    population = randomPopulation();
    EvRealVectorIndividual best  = 
      (EvRealVectorIndividual) this.getBestResult();
    EvRealVectorIndividual best2 = 
      (EvRealVectorIndividual) this.getSecondBestResult();
    EvRealVectorIndividual worst = 
      (EvRealVectorIndividual) this.getWorstResult();

    // Iteration over bits in mean vector 
    for (int k = 0; k < mean_vector.length; k++) {
      // Change mean_vector[k] by learning
      mean_vector[k] = mean_vector[k] * (1 - learning_rate)
             + (best.getValue(k) + best2.getValue(k) 
             - worst.getValue(k)) * learning_rate;
      
      //Change deviation_vector by learning (option D?)
//      deviation_vector[k] = deviation_vector[k] * (1 - learning_rate) 
//      + learning_rate * (Math.sqrt(sum/ k));
    }
    
    termination_condition.changeState(null);
  }

  /**
   * Gets the best individual in current population.
   * @return best individual
   */
  @Override
  public EvRealVectorIndividual getBestResult() {
    EvRealVectorIndividual best = population[0];
    
    for (int i = 1; i < population.length; i++) {
      if (objective_function.evaluate(best)   
          < objective_function.evaluate(population[i])) {
        best = population[i];
      }

    }
    return best;
  }
  
  /**
   *  Gets the second best individual in current population.
   *  @return second best individual
   */
  private EvRealVectorIndividual getSecondBestResult()  {

    EvRealVectorIndividual best = getBestResult();
    EvRealVectorIndividual best2 = best;

    for (int i = 0; i < population.length; i++)  {
      if (objective_function.evaluate(best)  
          > objective_function.evaluate(population[i])) {
        best2 = population[i];
        break;
      }
    }

    for (int i = 0; i < population.length; i++)  {
      if (objective_function.evaluate(best2)  
          < objective_function.evaluate(population[i]) 
          && objective_function.evaluate(best)  
          >= objective_function.evaluate(population[i])
          && best != population[i]) {
        best2 = population[i];
      }
    }
    
    return best2;
  }
  /**
   * Gets the worst individual in current population.
   * @return the worst individual in current population
   */
  @Override
  public EvRealVectorIndividual getWorstResult()  {
    EvRealVectorIndividual worst = population[0];

    for (int i = 1; i < population.length; i++)  {
      if (objective_function.evaluate(worst) 
          > objective_function.evaluate(population[i]))  {
        worst = population[i];
      }
    }
    return worst;
  }
  
  /**
   * Generate random population.
   * @return random population
   */
  private EvRealVectorIndividual[] randomPopulation() {
    EvRealVectorIndividual[] pop = new EvRealVectorIndividual[population_size];

    // Adding individual to population
    for (int i = 0; i < population_size; i++) {
      pop[i] = (EvRealVectorIndividual) solution_space
          .generateIndividual(mean_vector, deviation_vector);
    }

    return pop;
  }

  /**
   * Initial mean vector.
   * @param center Expected values vector (center of solution space)
   */
  private void initialMeanVector(final double[] center) {
    mean_vector = new double[center.length];
    for (int i = 0; i < mean_vector.length; i++) {
      mean_vector[i] = center[i];
    }
  }
  
  /**
   * Initial deviation vector.
   * @param deviation Standard deviations vector
   */
  private void initialDeviationVector(final double[] deviation)  {
    deviation_vector = new double[deviation.length];
    for (int i = 0; i < deviation_vector.length; i++)  {
      deviation_vector[i] = deviation[i];
    }
  }
  
}
