package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.totaltime;

import java.util.Date;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * In future it could be usefull for others. It will be tested in not too
 * distant future.
 * 
 * @author Tomasz Kozakiewicz
 */
public class EvTotalComputationTimeGatherer<T extends EvIndividual> extends
    EvGatherer<T> {

  public EvTotalComputationTimeGatherer(EvPersistentStatisticStorage storage) {
    this.setStorage(storage);
  }


  @Override
  public EvStatistic generate(EvPopulation<T> population) {

    Date node_current_time = new Date();
    return new EvTotalComputationTimeStatistic(node_current_time.getTime());
  }
}
