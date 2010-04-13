package pl.wroc.uni.ii.evolution.engine.operators.general.statistic;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * Abstract class for all gatherers which are special kind of operators. <br>
 * Their main purpose is to generate EvStatistic at every iteration and store it
 * using EvPersistentStatisticStorage.
 * 
 * @author Marcin Golebiowski
 * @param <T>
 */
public abstract class EvGatherer<T extends EvIndividual> implements
    EvOperator<T> {

  private EvPersistentStatisticStorage storage;


  public EvPopulation<T> apply(EvPopulation<T> population) {
    storage.saveStatistic(generate(population));
    return population;
  }


  public EvPersistentStatisticStorage getStorage() {
    return storage;
  }


  public void setStorage(EvPersistentStatisticStorage storage) {
    this.storage = storage;
  }


  /**
   * Returns some EvStatistic about a given population
   * 
   * @param population
   * @return EvStatistic
   */
  public abstract EvStatistic generate(EvPopulation<T> population);

}
