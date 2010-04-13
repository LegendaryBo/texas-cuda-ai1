package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.naturalnumbervector.statistic;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.avggenesvalue.EvGenesAvgValueGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.avggenesvalue.EvGenesAvgValueStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;

/**
 * Operator which gather information of average values of selected genes in a
 * natural vector individual - population
 * 
 * @author Kacper Gorski
 * @param <T>
 */
public class EvNaturalNumberVectorAvgGenesStat extends
    EvGenesAvgValueGatherer<EvNaturalNumberVectorIndividual> {

  public EvNaturalNumberVectorAvgGenesStat(int number_of_genes,
      EvPersistentStatisticStorage storage) {
    super(number_of_genes, storage);
  }


  /**
   * Build and return EvAvgGenesValueStatistic object made of genes of natural
   * vector individuals
   */
  public EvGenesAvgValueStatistic contructDataFromPopulation(
      EvPopulation population) {

    // suming genes and dividing by number of individuals
    for (Object individual : population) {
      for (int i = 0; i < number_of_genes; i++) {
        try {
          avg_values[i] +=
              ((EvNaturalNumberVectorIndividual) individual)
                  .getNumberAtPosition(i);

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
