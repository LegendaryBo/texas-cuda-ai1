package pl.wroc.uni.ii.evolution.distribution.statistics.persistency;

import java.io.IOException;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunication;

/**
 *
 */
public class EvPersistentStatisticServletStorage implements
    EvPersistentStatisticStorage {

  private long task_id;

  private long node_id;

  private long cell_id;

  private int iteration = 1;

  private EvDBServletCommunication comm;


  public EvPersistentStatisticServletStorage(Long task_id, Long cell_id,
      Long node_id, EvDBServletCommunication comm) {
    this.comm = comm;
    this.task_id = task_id;
    this.node_id = node_id;
    this.cell_id = cell_id;
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
    } catch (IOException e) {
      e.printStackTrace(System.out);
    }

  }

}
