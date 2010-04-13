package pl.wroc.uni.ii.evolution.engine.samplealgorithms;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * Sample implementation of CGA for BinaryStrings. Shows how to extend
 * EvolutionaryAlgorithm in an easy manner. Note: main computation is done in
 * doIteration method which performs only a <b>single</b> iteration of the
 * algorithm. This is because between consecutive iterations of the algorithm a
 * synchronization between distincd computing nodes will be performed.
 * 
 * @author Marcin Brodziak
 */
public final class EvCGA extends EvAlgorithm<EvBinaryVectorIndividual> {
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
   * Learning rate
   */
  private double learning_rate;


  /**
   * @param learning_rate
   */
  public EvCGA(double learning_rate) {
    super(2);
    this.learning_rate = learning_rate;
  }


  /**
   * Standard accessor, but overriden for BinaryStrings.
   */
  public void setSolutionSpace(EvSolutionSpace space) {
    solution_space = (EvBinaryVectorSpace) space;
  }


  /**
   * Initial probability distribution.
   */
  public void init() {
    initialProbabilityVector(solution_space.getDimension());
  }


  /**
   * A single iteration of CGA algorithm.
   */
  public void doIteration() {
    /*
     * Generation of two individuals with apropriate objective function set.
     * This is not very convenient that a user has to specify objective function
     * each time individual is constructor, but this is a design choice as
     * sometimes objective function changes over time.
     */
    individual1 =
        (EvBinaryVectorIndividual) solution_space
            .generateIndividual(probability_vector);
    individual1.setObjectiveFunction(objective_function);

    individual2 =
        (EvBinaryVectorIndividual) solution_space
            .generateIndividual(probability_vector);
    individual2.setObjectiveFunction(objective_function);

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
    for (int i = 0; i < solution_space.getDimension(); i++) {
      if (best_individual.getGene(i) == 1 && worst_individual.getGene(i) == 0) {
        probability_vector[i] = probability_vector[i] + learning_rate;
        if (probability_vector[i] > 1) {
          probability_vector[i] = 1;
        }
      } else if (best_individual.getGene(i) == 0
          && worst_individual.getGene(i) == 1) {
        probability_vector[i] = probability_vector[i] - learning_rate;
        if (probability_vector[i] < 0) {
          probability_vector[i] = 0;
        }
      }
    }

    /*
     * Termination condition is updated a little.
     */
    termination_condition.changeState(null);
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


  @Override
  public EvBinaryVectorIndividual getBestResult() {
    if (best_individual == null) {
      return null;
    }
    return best_individual;
  }
}
