package pl.wroc.uni.ii.evolution.distribution.clustering;

import java.io.IOException;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunication;

/**
 * Manage loading solution space for given cell from managment servlet.
 * 
 * @author Marcin Golebiowski
 */
public class EvSolutionSpaceLoaderFromServlet implements EvSolutionSpaceLoader,
    Runnable {

  private EvDBServletCommunication gateway;

  private long task_id;

  private long cell_id;

  private int delay;

  private int current_version = 0;

  private volatile EvSolutionSpace current_space = null;


  /**
   * Constructs object for given cell (<code> cell_id </code>) and task(<code> task_id </code>).
   * It communicates with servlet using given <code> gateway </code>
   * 
   * @param task_id
   * @param cell_id
   * @param gateway
   * @param delay
   */
  public EvSolutionSpaceLoaderFromServlet(long task_id, long cell_id,
      EvDBServletCommunication gateway, int delay) {

    this.task_id = task_id;
    this.cell_id = cell_id;
    this.delay = delay;
    this.gateway = gateway;

  }


  /**
   * Loads the most fresh solution space
   */
  public EvSolutionSpace takeSubspace() {
    EvSolutionSpace ss_to_return = current_space;
    current_space = null;
    return ss_to_return;
  }


  /**
   * Check if there is a new solution space, but it connects to managment
   * servlet only after given delay
   * 
   * @return <code> true </code> if there is a new solution space <br />
   *         <code> false </code> if there isn't a new solution space <br />
   */
  public boolean newSubspaceAvailable() {
    return current_space != null;
  }


  public void run() {
    for (;;) {
      try {
        update();
        Thread.sleep(delay);
      } catch (InterruptedException e) {
        return;
      } catch (Exception ex) {
        throw new RuntimeException(ex);
      }
    }
  }


  /**
   * Updates connects with servlet to update its solution subspace. Intended for
   * internal use only; it is invoked by run every delay miliseconds.
   * 
   * @throws IOException
   */
  public void update() throws IOException {

    int database_version =
        gateway.getVersionOfNewSolutonSpace(task_id, cell_id);
    if (database_version > current_version) {
      current_space =
          (EvSolutionSpace) gateway.getSolutionSpace(task_id, cell_id);
      current_version = database_version;
    }
  }


  public Thread start() {
    Thread worker = new Thread(this);
    worker.start();
    return worker;
  }
}
