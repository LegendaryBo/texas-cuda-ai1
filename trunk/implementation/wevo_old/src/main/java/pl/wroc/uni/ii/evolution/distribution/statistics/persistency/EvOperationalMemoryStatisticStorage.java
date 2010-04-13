package pl.wroc.uni.ii.evolution.distribution.statistics.persistency;

import java.util.ArrayList;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;

/**
 * Objects storing statistics in local operational memory.
 * 
 * @author Kacper Gorski
 */
public class EvOperationalMemoryStatisticStorage implements
    EvPersistentStatisticStorage {

  ArrayList<EvStatistic> statistics = new ArrayList<EvStatistic>();


  public EvStatistic[] getStatistics() {
    return statistics.toArray(new EvStatistic[statistics.size()]);
  }


  public void saveStatistic(EvStatistic stats) {
    statistics.add(stats);
  }

}
