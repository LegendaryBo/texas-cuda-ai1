package pl.wroc.uni.ii.evolution.engine.prototype.operators;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;

/**
 * This interface notifies, that class implementing it can gather statistics
 * about bayesian network it uses. <br>
 * Class implementing this interface should start gathering statistics only if
 * method collectBayesianStats was called. Every call of apply should call
 * saveStatistic of the storage object. Further statistics can be viewed using
 * data stored in storage object given in collectBayesianStats method and
 * wisualised by EvBayesianNeyworkChart class.
 * 
 * @author Kacper Gorski (admin@34all.org)
 */
public interface EvBayesianOperator {

  /**
   * Method notifies operator to start collecting data into the given storage.
   * 
   * @param storage in which information about bayesian network ill be stored.
   */
  void collectBayesianStats(EvPersistentStatisticStorage storage);

}
