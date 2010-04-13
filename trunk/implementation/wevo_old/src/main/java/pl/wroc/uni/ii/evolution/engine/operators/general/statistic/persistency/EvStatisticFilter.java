package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;

/**
 * Usufull filtering methods for EvStatistics
 * 
 * @author Marcin Golebiowski
 */
public class EvStatisticFilter {

  @SuppressWarnings("unchecked")
  public static EvStatistic[] byClass(Class statistic_class, EvStatistic[] stats) {

    if (stats == null) {
      return null;
    }

    int count = 0;
    for (int i = 0; i < stats.length; i++) {
      if (stats[i].getClass().equals(statistic_class)) {
        count++;
      }
    }

    if (count == 0) {
      return null;
    } else {
      EvStatistic[] result = new EvStatistic[count];

      int j = 0;
      for (int i = 0; i < stats.length; i++) {
        if (stats[i].getClass().equals(statistic_class)) {
          result[j++] = stats[i];
        }
      }

      return result;
    }
  }


  @SuppressWarnings("unchecked")
  public static EvStatistic[] byClassAndIteration(Class statistic_class,
      EvStatistic[] stats, int start_iteration, int end_iteration) {

    if (stats == null) {
      return null;
    }

    int count = 0;
    for (int i = 0; i < stats.length; i++) {
      if ((stats[i].getClass().equals(statistic_class))
          && (stats[i].getIteration() >= start_iteration)
          && (stats[i].getIteration() <= end_iteration)) {
        count++;
      }
    }

    if (count == 0) {
      return null;
    } else {
      EvStatistic[] result = new EvStatistic[count];

      int j = 0;
      for (int i = 0; i < stats.length; i++) {
        if ((stats[i].getClass().equals(statistic_class))
            && (stats[i].getIteration() >= start_iteration)
            && (stats[i].getIteration() <= end_iteration)) {
          result[j++] = stats[i];
        }
      }
      return result;
    }
  }
}
