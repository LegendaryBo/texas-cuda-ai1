package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;

/**
 * An interface for all object that provides persistent storage for EvStatitic
 * objects.
 * 
 * @author Marcin Golebiowski
 */
public interface EvPersistentStatisticStorage {

  /**
   * Add <b> stats </b> to persistent storage
   * 
   * @param stats
   */
  public void saveStatistic(EvStatistic stats);


  /**
   * Returns all stored EvStatistic objects.
   * 
   * @return array of EvStatistic
   */
  public EvStatistic[] getStatistics();
}
