package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Gathers EvMaxAvgMinStatistic for evolution.
 * 
 * @author Marcin Golebiowski
 * @param <T>
 */
public class EvObjectiveFunctionValueMaxAvgMinGatherer<T extends EvIndividual>
    extends EvGatherer<T> {

  private int iteration = 0;


  public EvObjectiveFunctionValueMaxAvgMinGatherer(
      EvPersistentStatisticStorage storage) {
    this.setStorage(storage);
  }


  @Override
  public EvStatistic generate(EvPopulation<T> population) {

    double avg = 0.0;
    for (T i : population) {
      avg += i.getObjectiveFunctionValue();
    }

    avg /= population.size();

    double stddev = 0.0;
    for (T i : population) {
      stddev +=
          (i.getObjectiveFunctionValue() - avg)
              * (i.getObjectiveFunctionValue() - avg);
    }

    stddev = Math.sqrt(stddev / population.size());

    return new EvObjectiveFunctionValueMaxAvgMinStatistic(iteration++,
        population.getBestResult().getObjectiveFunctionValue(), avg, population
            .getWorstResult().getObjectiveFunctionValue(), stddev);
  }
}
