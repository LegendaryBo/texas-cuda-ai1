package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency;

import java.util.ArrayList;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;

/**
 * Simple storage object that stores data in virtual memory.<br>
 * 
 * @author Kacper Gorski (admin@34all.org)
 */
public class EvPersistentSimpleStorage implements EvPersistentStatisticStorage {

  /**
   * EvStatistics storage.
   */
  private ArrayList<EvStatistic> storage = new ArrayList<EvStatistic>();


  /**
   * @param stats EvStatistic object to be added
   */
  public void saveStatistic(final EvStatistic stats) {
    storage.add(stats);
  }


  /**
   * @return array of EvStatistics objects, first element point to and object
   *         from first iteration.
   */
  public EvStatistic[] getStatistics() {
    return storage.toArray(new EvStatistic[storage.size()]);
  }

}
