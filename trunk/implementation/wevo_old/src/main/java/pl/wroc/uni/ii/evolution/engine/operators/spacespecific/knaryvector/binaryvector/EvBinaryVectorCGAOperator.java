package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * Sample implementation of CGA operator for BinaryStrings. Note: main
 * computation is done in apply method which performs only a <b>single</b>
 * iteration of the algorithm. This is because between consecutive iterations of
 * the algorithm a synchronization between distincd computing nodes will be
 * performed.
 * 
 * @author Marcin Golebiowski, Kacper Gorski Profiled 14.V.2007
 */
public class EvBinaryVectorCGAOperator implements
    EvOperator<EvBinaryVectorIndividual> {

  /**
   * population size of generated population by this operator
   */
  private int population_size;

  /**
   * indicated how much the vector of probability changes when mutated
   */
  private double learning_rate;

  /**
   * number of iteration used to generate new opulation
   */
  private int iterations;

  /**
   * Solution space for which CGA works.
   */
  private EvBinaryVectorSpace solution_space;

  /**
   * Two individuals, four variables for clarity.
   */
  private EvBinaryVectorIndividual individual1, individual2, best_individual,
      worst_individual;

  /**
   * Probability vector that works is used throughout CGA.
   */
  private double probability_vector[];


  /**
   * Default constructor of CGA operator.
   * 
   * @param objective_function_ - objective function used to evaluate binary
   *        individuals
   * @param learning_rate_ - number of iterations before generating population
   *        with apply().
   * @param learning_rate_ - how much muttation affects vector of probability.
   * @param population_size_ - size of populaton generated from vector of
   *        probability
   * @param solution_space_ - solution space
   */
  public EvBinaryVectorCGAOperator(int iterations, double learning_rate_,
      int population_size_, EvBinaryVectorSpace solution_space_) {

    if (population_size_ <= 0) {
      throw new IllegalArgumentException(
          "Population size must be a positive Integer");

    }
    if (learning_rate_ < 0 || learning_rate_ > 1) {
      throw new IllegalArgumentException(
          "Learing rate must be a Double in range[0,1]");
    }

    if (solution_space_ == null) {
      throw new IllegalArgumentException("Solution_space");
    }

    if (iterations <= 0) {
      throw new IllegalArgumentException(
          "Iterations must be a positive Integer");
    }

    this.iterations = iterations;
    learning_rate = learning_rate_;
    population_size = population_size_;
    setSolutionSpace(solution_space_);
    init();

  }


  /* TODO we don't use variable population in function header!! What to do? */
  /**
   * Generates population of given size in the constructor from probability
   * vector evaluated durning previous iterations.<BR>
   * NOTE: Argument population is not used and should be set to null
   */
  public EvPopulation<EvBinaryVectorIndividual> apply(
      EvPopulation<EvBinaryVectorIndividual> population) {
    /*
     * Generation of two individuals with apropriate objective function set.
     * This is not very convenient that a user has to specify objective function
     * each time individual is constructor, but this is a design choice as
     * sometimes objective function changes over time.
     */

    for (int j = 0; j < iterations; j++) {

      individual1 =
          (EvBinaryVectorIndividual) solution_space
              .generateIndividual(probability_vector);
      individual1.setObjectiveFunction(solution_space.getObjectiveFuntion());

      individual2 =
          (EvBinaryVectorIndividual) solution_space
              .generateIndividual(probability_vector);
      individual2.setObjectiveFunction(solution_space.getObjectiveFuntion());

      /*
       * Choosing better and worse of the two.
       */
      if (individual1.getObjectiveFunctionValue() > individual2
          .getObjectiveFunctionValue()) {
        best_individual = individual1;
        worst_individual = individual2;
      } else {
        best_individual = individual2;
        worst_individual = individual1;
      }

      /*
       * We update each position in probabilty vector by a theta factor in an
       * apropriate way if individuals differ on a given position.
       */

      int solution_space_dimension = solution_space.getDimension();
      for (int i = 0; i < solution_space_dimension; i++) {
        if ((best_individual.getGene(i) == 1)
            && (worst_individual.getGene(i) == 0)) {
          probability_vector[i] = probability_vector[i] + learning_rate;
          if (probability_vector[i] > 1) {
            probability_vector[i] = 1;
          }
        } else if ((best_individual.getGene(i) == 0)
            && (worst_individual.getGene(i) == 1)) {
          probability_vector[i] = probability_vector[i] - learning_rate;
          if (probability_vector[i] < 0) {
            probability_vector[i] = 0;
          }
        }
      }

    }
    return randomPopulation();
  }


  /**
   * Standard accessor, but overriden for BinaryStrings.
   */
  public void setSolutionSpace(EvSolutionSpace space) {
    solution_space = (EvBinaryVectorSpace) space;
  }


  /*
   * generating random population from probability vector
   */
  private EvPopulation<EvBinaryVectorIndividual> randomPopulation() {
    EvPopulation<EvBinaryVectorIndividual> pop =
        new EvPopulation<EvBinaryVectorIndividual>();

    // Adding individual to population
    for (int i = 0; i < population_size; i++) {
      pop.add((EvBinaryVectorIndividual) solution_space
          .generateIndividual(probability_vector));
    }

    return pop;
  }


  /*
   * initializing default probability vector
   */
  private void init() {
    initialProbabilityVector(solution_space.getDimension());
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

}
