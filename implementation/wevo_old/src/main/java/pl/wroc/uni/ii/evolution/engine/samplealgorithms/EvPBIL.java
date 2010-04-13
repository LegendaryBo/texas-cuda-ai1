package pl.wroc.uni.ii.evolution.engine.samplealgorithms;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Sample and simple implementation of PBIL for BinaryStrings.
 * 
 * @author Marcin Golebiowski i Kacper Górski
 */
public final class EvPBIL extends EvAlgorithm<EvBinaryVectorIndividual> {

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
   * Some population for learning probablity vector
   */
  private EvBinaryVectorIndividual population[];


  /**
   * @param theta1 learning rate
   * @param theta2 mutation probability
   * @param theta3 mutation rate
   * @param population_size
   */
  public EvPBIL(double theta1, double theta2, double theta3, int population_size) {
    super(population_size);
    this.theta1 = theta1;
    this.theta2 = theta2;
    this.theta3 = theta3;
  }


  /**
   * Initial probability distribution.
   */
  public void init() {
    initialProbabilityVector(solution_space.getDimension());
  }


  /**
   * Standard accessor, but overriden for BinaryStrings.
   */
  public void setSolutionSpace(EvSolutionSpace<EvBinaryVectorIndividual> space) {
    solution_space = (EvBinaryVectorSpace) space;
  }


  /**
   * A single iteration for PBIL
   */
  @Override
  public void doIteration() {

    population = randomPopulation();
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

    termination_condition.changeState(null);
  }


  /**
   * Gets the best individual in current population
   */
  @Override
  public EvBinaryVectorIndividual getBestResult() {
    EvBinaryVectorIndividual best = population[0];

    for (int i = 1; i < population.length; i++) {
      if (objective_function.evaluate(best) < objective_function
          .evaluate(population[i])) {
        best = population[i];
      }

    }
    return best;
  }


  private EvBinaryVectorIndividual[] randomPopulation() {
    EvBinaryVectorIndividual pop[] = new EvBinaryVectorIndividual[n];

    // Adding individual to population
    for (int i = 0; i < n; i++) {
      pop[i] =
          (EvBinaryVectorIndividual) solution_space
              .generateIndividual(probability_vector);
    }

    return pop;
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
