package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.objectivefunctiondistr;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * It gathers the information about:
 * <ul>
 * <li> fitness values in a population
 * <li> how many individuals have that fitness value
 * </ul>
 * 
 * @author Marcin Golebiowski
 * @param <T>
 */
public class EvObjectiveFunctionDistributionGatherer<T extends EvIndividual>
    extends EvGatherer<T> {

  private int iteration = 0;


  /**
   * Constructor.
   * 
   * @param storage -- stores statistics
   */
  public EvObjectiveFunctionDistributionGatherer(
      EvPersistentStatisticStorage storage) {
    this.setStorage(storage);
  }


  /**
   * Returns EvStatistic about a given population
   */
  @Override
  public EvStatistic generate(EvPopulation<T> population) {

    EvObjectiveFunctionValueDistributionStatistic result =
        new EvObjectiveFunctionValueDistributionStatistic(iteration++);

    /** sort indviduals in the population using their fitness values * */
    population.sort();

    Double prev = null;
    int count = 0;

    /** iterate over individuals in the population * */
    for (EvIndividual individual : population) {

      if (prev != null) {
        if (individual.getObjectiveFunctionValue() == prev) {
          count++;
        } else {
          result.addFinesssValueStatistic(new EvObjectiveValueStatistic(prev,
              count));
          count = 0;
          prev = null;
        }
      }

      if (prev == null) {
        prev = individual.getObjectiveFunctionValue();
        count = 1;
      }
    }

    if (count != 0) {
      result
          .addFinesssValueStatistic(new EvObjectiveValueStatistic(prev, count));
    }

    return result;
  }

}
