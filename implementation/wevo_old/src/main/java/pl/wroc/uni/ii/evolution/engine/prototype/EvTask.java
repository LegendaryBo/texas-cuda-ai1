package pl.wroc.uni.ii.evolution.engine.prototype;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;

/**
 * Encapsulates all information about a given evolutionary task, that means:
 * algorithm, objective function, all necessary parameters.
 * 
 * @author Marcin Brodziak (marcin@nierobcietegowdomu.pl)
 * @author Tomasz Kozakiewicz (quzzaq@gmail.com)
 * @author Marcin Golebiowski (xormus@gmail.com)
 */
public class EvTask implements Runnable {

  /**
   * Algorithm for the task.
   */
  protected EvAlgorithm task_algorithm = null;


  /**
   * Sets an evolutionary algorithm for this task.
   * 
   * @param algorithm -- algorithm to use
   */
  @SuppressWarnings("unchecked")
  public void setAlgorithm(final EvAlgorithm algorithm) {

    if (algorithm == null) {
      throw new IllegalArgumentException("You have called a setAlgoritm "
          + "function with the algorithm equals to null");
    }

    this.task_algorithm = algorithm;
  }


  /**
   * Methods that runs algorithm basing on termination condition.
   */
  @SuppressWarnings( {"unchecked", "deprecation"})
  public void run() {

    if (task_algorithm == null) {
      throw new IllegalStateException(
          "Run method of EvolutionaryTask required to call "
              + "setAlgorithm(EvolutionaryAlgorithm) first.");
    }

    // initialize algorithm
    task_algorithm.init();
    while (!task_algorithm.isTerminationConditionSatisfied()) {
      // do one algorithm iteration
      task_algorithm.doIteration();
    }
  }


  /**
   * Prints best result.
   */
  public void printBestResult() {
    if (task_algorithm == null) {
      throw new IllegalStateException(
          "Print best result method of EvolutionaryTask required to call "
              + "setAlgorithm(EvolutionaryAlgorithm) first.");
    }
    if (task_algorithm.getBestResult() == null) {
      throw new IllegalStateException(
          "Print best result method of EvolutionaryTask required to call "
              + "run() first.");
    }
    System.out.println(task_algorithm.getBestResult());
  }


  /**
   * Returns the best individual in the current population.
   * 
   * @return individual
   */
  public EvIndividual getBestResult() {
    if (task_algorithm == null) {
      throw new IllegalStateException(
          "Get best result method of EvolutionaryTask required to "
              + "call setAlgorithm(EvolutionaryAlgorithm) first.");
    }
    if (task_algorithm.getBestResult() == null) {
      throw new IllegalStateException(
          "Get best result method of EvolutionaryTask required "
              + "to call run() first.");
    }

    return task_algorithm.getBestResult();
  }
}
