package pl.wroc.uni.ii.evolution.distribution.tasks;

import pl.wroc.uni.ii.evolution.distribution.clustering.EvSolutionSpaceLoader;
import pl.wroc.uni.ii.evolution.distribution.eval.EvExternalEvaluationOperator;
import pl.wroc.uni.ii.evolution.distribution.strategies.EvIslandModel;
import pl.wroc.uni.ii.evolution.distribution.workers.EvBlankEvolInterface;
import pl.wroc.uni.ii.evolution.distribution.workers.EvEvolutionInterface;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;

/**
 * Encapsulates all information about a given evolutionary task, that means:
 * algorithm, objective function, all necessary parameters.
 * 
 * @author Marcin Brodziak
 * @author Tomasz Kozakiewicz
 * @author Marcin Golebiowski (xormus@gmail.com)
 */
public class EvIslandDistribution extends EvTask {

  private EvIslandModel strategy = null;

  private EvSolutionSpaceLoader loader = null;

  private Thread loader_thread = null;

  private EvExternalEvaluationOperator<EvIndividual> eval = null;

  private EvEvolutionInterface inter = null;


  /**
   * Creates EvIsland task object. It is necessary to call
   * setDistributedStrategy and setAlgorithm before running this task
   */
  public EvIslandDistribution() {

  }


  /**
   * Creates EvIsland task with given island model and algorithm
   * 
   * @param strategy
   * @param algorithm
   */
  public EvIslandDistribution(EvIslandModel strategy, EvAlgorithm algorithm) {
    setDistributedStrategy(strategy);
    setAlgorithm(algorithm);
  }


  /**
   * Sets strategy for exchanging individuals
   * 
   * @param strategy
   */
  public void setDistributedStrategy(EvIslandModel strategy) {
    this.strategy = strategy;
  }


  /**
   * Set the way the task loads new SolutionSpace
   * 
   * @param loader
   */
  public void setSolutionSpaceLoader(EvSolutionSpaceLoader loader) {
    this.loader = loader;
  }


  @SuppressWarnings("unchecked")
  public void setExternalEval(EvExternalEvaluationOperator operator) {
    this.eval = operator;
  }


  /**
   * Methods that runs algorithm basing on termination condition.
   */
  @Override
  @SuppressWarnings( {"unchecked", "deprecation"})
  public void run() {

    if (inter == null)
      inter = new EvBlankEvolInterface();

    task_algorithm.setInterface(inter);
    inter.newTask();
    inter.currentCellID(strategy.getSender().getTopology().assignCellID());

    if (strategy != null) {
      inter.currentTask((int) strategy.getSender().getTaskID());
      inter.currentNode((int) strategy.getSender().getNodeID());
    }

    if (task_algorithm == null) {
      throw new IllegalStateException(
          "Run method of EvolutionaryTask required to call setAlgorithm(EvolutionaryAlgorithm) first.");
    }

    // init algorithm
    task_algorithm.init();

    // init distribution strategy
    if (strategy != null) {
      strategy.init(inter);
    }

    // starts loader
    if (loader != null) {
      loader_thread = loader.start();
    }

    while (!task_algorithm.isTerminationConditionSatisfied()) {

      // check if thread is interrupted
      if (Thread.interrupted()) {
        break;
      }

      long start_time = System.currentTimeMillis();
      task_algorithm.doIteration(); // do one algorithm iteration
      long end_time = System.currentTimeMillis();

      if (strategy != null) {

        if (!strategy.isOkey()) {
          System.out.println("Problem with strategy");
          break;
        }

        if (eval != null) {
          task_algorithm.setPopulation(eval.apply(task_algorithm
              .getPopulation()));
        }

        // System.out.println("Synchronize");
        // synchronize with neighbours
        strategy.export(task_algorithm.getPopulation());
        strategy.updatePopulation(task_algorithm.getPopulation());
      }

      // send details to interface
      inter.iterationProgress();
      inter.addEvalTime((int) (end_time - start_time));
      inter.currentPopulation((task_algorithm.getPopulation().clone()));
      inter.populationSize(task_algorithm.getPopulation().size());

      if (loader != null && loader.newSubspaceAvailable()) {
        // load solution space from somewhere
        task_algorithm.setSolutionSpace(loader.takeSubspace());

        // init algorithm
        task_algorithm.init();

        // stops current strategy
        if (strategy != null) {
          strategy.stop();
        }

        // init new distribution strategy
        if (strategy != null) {
          strategy.init(inter);
        }
      }
    }

    // stopping strategy
    if (strategy != null) {
      strategy.stop();
    }

    // stopping loader
    if (loader != null) {
      loader_thread.stop();
    }
  }


  /**
   * If nothing is set, blank interface is used.
   * 
   * @param inter - Inteface object which will be informed what is happening
   *        inside evolution
   */
  public void setInterface(EvEvolutionInterface inter) {
    this.inter = inter;
  }

}
