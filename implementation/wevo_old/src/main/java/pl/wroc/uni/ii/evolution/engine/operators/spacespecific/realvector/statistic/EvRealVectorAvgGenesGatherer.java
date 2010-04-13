package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector.statistic;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.avggenesvalue.EvGenesAvgValueGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.avggenesvalue.EvGenesAvgValueStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;

/**
 * @author Kacper Gorski
 */
public class EvRealVectorAvgGenesGatherer extends
    EvGenesAvgValueGatherer<EvRealVectorIndividual> {

  public EvRealVectorAvgGenesGatherer(int number_of_genes,
      EvPersistentStatisticStorage storage) {
    super(number_of_genes, storage);
  }


  /**
   * Build and return EvAvgGenesValueStatistic object made of genes of real
   * vector individuals
   */
  public EvGenesAvgValueStatistic contructDataFromPopulation(
      EvPopulation population) {

    // suming genes values and dividing by number of individuals
    for (Object individual : population) {
      for (int i = 0; i < number_of_genes; i++) {
        try {
          avg_values[i] += ((EvRealVectorIndividual) individual).getValue(i);

        } catch (NullPointerException e) {
          throw new IllegalStateException("Index " + i
              + " is behind the range of the individual");
        }
      }
    }

    for (int i = 0; i < number_of_genes; i++) {
      avg_values[i] = avg_values[i] / ((double) population.size());
    }

    return new EvGenesAvgValueStatistic(avg_values, iteration);
  }

}
