package pl.wroc.uni.ii.evolution.distribution.workers;

import java.util.List;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * Interface that allows to communicate beetwen evolutionary algorithm and
 * application. It's methods are called by evolutionary tasks, which let know
 * the application implementing this interface what's going on inside the task.
 * Currently implemented by EvEvolutionApplet and EvConsoleWorker
 * 
 * @author Kacper Gorski
 */
public interface EvEvolutionInterface {

  /**
   * Notes about single iteration progress
   */
  public void iterationProgress();


  /**
   * Gives the task id of currently evolution task
   * 
   * @param task_id
   */
  public void currentTask(int task_id);


  /**
   * Gives the node id of currently evolution task
   * 
   * @param node_id
   */
  public void currentNode(int node_id);


  /**
   * Returns poopulation size of currently evolution task
   * 
   * @param node_id
   */
  public void populationSize(int size);


  /**
   * Inform about the ammount of time since previous call of this method
   * 
   * @param time
   */
  public void addEvalTime(int time);


  /**
   * Inform about the ammount of time spent on applying specified operator
   * 
   * @param operator whose time was measured
   * @param time spent on evaluation above operator
   */
  public void addOperatortime(EvOperator operator, int time);


  /**
   * @param individuals imported recently from neighborhood nodes to current
   *        task
   */
  public void addImportedPopulation(List individuals);


  /**
   * @param individuals exported recently to wEvo server
   */
  public void addExportedPopulation(List individuals);


  /**
   * @param population of current evolutionary task
   */
  public void currentPopulation(EvPopulation population);


  /**
   * Notification that new task has started
   */
  public void newTask();


  /**
   * Gives the cell id of currently evolution task
   * 
   * @param cell_id
   */
  public void currentCellID(long cell_id);

}
