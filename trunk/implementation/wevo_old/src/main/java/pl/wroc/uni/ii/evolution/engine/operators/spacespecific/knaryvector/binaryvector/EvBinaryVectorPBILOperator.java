package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Simple implementation of PBIL operator. <BR>
 * Operator at the beggining creates default probability vector which is used to
 * generate individuals in futher evaluation. Every apply of this operator
 * improves probability vector and returns a new population generated from
 * current vector. See EvPBILExample in sampleimplementation packet.
 * 
 * @author Marcin Golebiowski i Kacper Gorski
 */
public class EvBinaryVectorPBILOperator implements
    EvOperator<EvBinaryVectorIndividual> {

  /**
   * Solution space for which CGA works.
   */
  private EvBinaryVectorSpace solution_space;

  /**
   * Learning rate
   */
  private double theta1;

  /**
   * Mutation probability
   */
  private double theta2;

  /**
   * Mutation rate
   */

  private double theta3;

  /**
   * Probability vector that works is used throughout CGA.
   */
  private double probability_vector[];

  /**
   * Population size
   */
  private int n;

  /**
   * number of iteration used to generate new opulation
   */
  private int iterations;

  /**
   * Some population for learning probablity vector
   */
  private EvPopulation<EvBinaryVectorIndividual> population;

  /**
   * function used to evaluate binary individuals in this operator
   */
  private EvObjectiveFunction<EvBinaryVectorIndividual> objective_function;


  /**
   * Default contructor for PBIL operator.
   * 
   * @param objective_function_ - objective function which evaluate binary
   *        inidivduals
   * @param iterations - number of iterations after which the new populations is
   *        given
   * @param learning_rate - learning rate of the operator.<BR>
   *        Value must be <code> Double </code> in range <code> [0,1] </code>
   * @param mutation_probability - chance of random change in individuals.<BR>
   *        Value must be <code> Double </code> in range <code> [0,1] </code>
   * @param mutation_rate - how much the probability changes after evaluation.<BR>
   *        Value must be <code> Double </code> in range <code> [0,1] </code>
   * @param population_size - size of the evaluated population.<BR>
   *        Value must be positive <code> Integer </code>
   * @param solution_space - ???
   */
  public EvBinaryVectorPBILOperator(
      EvObjectiveFunction<EvBinaryVectorIndividual> objective_function_,
      int iterations, double learning_rate, double mutation_probability,
      double mutation_rate, int population_size,
      EvBinaryVectorSpace solution_space) {

    // check if we give correct values
    if (mutation_rate < 0 || mutation_rate > 1) {
      throw new IllegalArgumentException(
          "PBIL accepts only Mutation_RATE as a parameter"
              + " which must be a Double in range[0,1]");
    }
    theta3 = mutation_rate;
    if (mutation_probability < 0 || mutation_probability > 1) {
      throw new IllegalArgumentException(
          "PBIL accepts only Mutation_PROBABILITY as a parameter"
              + " which must be a Double in range[0,1]");
    }
    theta2 = mutation_probability;

    if (learning_rate < 0 || learning_rate > 1) {
      throw new IllegalArgumentException(
          "PBIL accepts only LEARNING_RATE as a parameter"
              + " which must be a Double in range[0,1]");
    }
    theta1 = learning_rate;

    if (population_size <= 0) {
      throw new IllegalArgumentException(
          "PBIL accepts only POPULATION_SIZE as a parameter"
              + " which must be a positive Integer");
    }
    n = population_size;

    if (iterations <= 0) {
      throw new IllegalArgumentException(
          "PBIL accepts only iterations as a parameter"
              + " which must be a positive Integer");
    }
    this.iterations = iterations;

    if (solution_space == null || objective_function_ == null) {
      throw new IllegalArgumentException(
          "solution_space and objective function cannot be null");
    }

    setSolutionSpace(solution_space);
    objective_function = objective_function_;
    init();

  }


  /**
   * set solution space
   */
  public void setSolutionSpace(EvSolutionSpace<EvBinaryVectorIndividual> space) {
    solution_space = (EvBinaryVectorSpace) space;
  }


  /**
   * Initial probability vector is set to 0.5.
   */
  private void initialProbabilityVector(int d) {
    probability_vector = new double[d];
    for (int i = 0; i < probability_vector.length; i++) {
      probability_vector[i] = 0.5;
    }
  }


  /**
   * Generates population of given size in the contructor from probability
   * vector evaluated durning previous iterations.<BR>. NOTE: Argument
   * population is not used and should be set to null
   */
  public EvPopulation<EvBinaryVectorIndividual> apply(
      EvPopulation<EvBinaryVectorIndividual> population) {
    /*
     * TODO this operator doesnt need population as argument
     */

    for (int i = 0; i < iterations; i++) {

      this.population = randomPopulation();
      EvBinaryVectorIndividual best =
          (EvBinaryVectorIndividual) this.getBestResult();

      // Iteration over bits in probability vector
      for (int k = 0; k < probability_vector.length; k++) {
        // Change probability_vector[k] by learning
        probability_vector[k] =
            probability_vector[k] * (1 - theta1)
                + (best.getGene(k) == 1 ? 1.0 : 0) * theta1;

        // Randomly choose if mutate or not mutate
        if (EvRandomizer.INSTANCE.nextDouble() < theta2) {
          // Change probability_vector[k] by mutation
          probability_vector[k] =
              probability_vector[k] * (1 - theta3)
                  + (EvRandomizer.INSTANCE.nextProbableBoolean(0.5) ? 1.0 : 0)
                  * theta3;
        }
      }

    }

    return this.population;
  }


  /**
   * Gets the best individual in current population
   */
  public EvBinaryVectorIndividual getBestResult() {
    EvBinaryVectorIndividual best = population.get(0);

    for (int i = 1; i < population.size(); i++) {
      if (objective_function.evaluate(best) < objective_function
          .evaluate(population.get(i))) {
        best = population.get(i);
      }

    }
    return best;
  }


  /*
   * initializing default probability vector
   */
  private void init() {
    initialProbabilityVector(solution_space.getDimension());
  }


  /*
   * generating random population from probability vector
   */
  private EvPopulation<EvBinaryVectorIndividual> randomPopulation() {
    EvPopulation<EvBinaryVectorIndividual> pop =
        new EvPopulation<EvBinaryVectorIndividual>();

    // Adding individual to population
    for (int i = 0; i < n; i++) {
      pop.add((EvBinaryVectorIndividual) solution_space
          .generateIndividual(probability_vector));
    }

    return pop;
  }

}
