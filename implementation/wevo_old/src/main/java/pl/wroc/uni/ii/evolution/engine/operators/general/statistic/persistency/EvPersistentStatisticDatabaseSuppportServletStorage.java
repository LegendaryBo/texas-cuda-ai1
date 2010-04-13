package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency;

import java.io.IOException;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunication;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunicationImpl;

/**
 * Object representing centralised database storage for saving statistics. Every
 * statistic object is saved on central wEvo database together with his taks_id,
 * cell_id, node_id and time. This object is only virtual storage, as all the
 * data is stored on central wevo computer
 * 
 * @author Marcin Golebiowski, Kacper Gorski
 */
public class EvPersistentStatisticDatabaseSuppportServletStorage implements
    EvPersistentStatisticStorage {

  private int task_id;

  private long node_id;

  private long cell_id;

  private int iteration = 1; // current iteration, starting from 1

  // database interface implementation
  private EvDBServletCommunication comm;


  /**
   * Created storage object
   * 
   * @param task_id - task_id of current algorithm
   * @param cell_id - cell_id of current algorithm
   * @param node_id - node_id of current algorithm
   * @param comm - database interface implementation
   */
  public EvPersistentStatisticDatabaseSuppportServletStorage(int task_id,
      long cell_id, long node_id, EvDBServletCommunication comm) {
    this.comm = comm;
    this.task_id = task_id;
    this.node_id = node_id;
    this.cell_id = cell_id;
  }


  /**
   * Created storage object without necceserity to create
   * EvDBServletCommunication
   * 
   * @param task_id - task_id of current algorithm
   * @param cell_id - cell_id of current algorithm
   * @param node_id - node_id of current algorithm
   * @param wevo_server_url
   */
  public EvPersistentStatisticDatabaseSuppportServletStorage(int task_id,
      long cell_id, long node_id, String wevo_server_url) {
    this(task_id, cell_id, node_id, new EvDBServletCommunicationImpl(
        wevo_server_url));
  }


  public EvStatistic[] getStatistics() {

    try {
      Object[] result = comm.getStatistics(task_id, cell_id, node_id);

      if (result == null) {
        return null;
      }
      EvStatistic[] stats = new EvStatistic[result.length];

      for (int i = 0; i < result.length; i++) {
        stats[i] = (EvStatistic) result[i];
      }

      return stats;

    } catch (IOException e) {
      e.printStackTrace(System.out);
    }

    return null;

  }


  public void saveStatistic(EvStatistic stats) {

    try {
      comm.saveStatistic(task_id, cell_id, node_id, stats, iteration);
      iteration++;
    } catch (IOException e) {
      e.printStackTrace(System.out);
    }

  }

}
