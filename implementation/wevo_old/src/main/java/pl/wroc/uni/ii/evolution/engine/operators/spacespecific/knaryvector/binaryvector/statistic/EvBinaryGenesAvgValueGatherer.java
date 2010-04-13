package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.statistic;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.avggenesvalue.EvGenesAvgValueGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.avggenesvalue.EvGenesAvgValueStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;

/**
 * Operator which gather information of average values of selected genes in a
 * BinaryIndividuals population
 * 
 * @author Kacper Górski
 */
public class EvBinaryGenesAvgValueGatherer extends
    EvGenesAvgValueGatherer<EvBinaryVectorIndividual> {

  /**
   * Contructs gathering operator which collect information about average genes
   * value.
   * 
   * @param number_of_genes
   * @param storage
   */
  public EvBinaryGenesAvgValueGatherer(int number_of_genes,
      EvPersistentStatisticStorage storage) {
    super(number_of_genes, storage);
  }


  /**
   * Build and return EvAvgGenesValueStatistic object made of genes of binary
   * string individuals
   */
  public EvGenesAvgValueStatistic contructDataFromPopulation(
      EvPopulation population) {

    // couting true genes and dividing by number of individuals
    for (Object individual : population) {
      for (int i = 0; i < number_of_genes; i++) {
        try {
          if (((EvBinaryVectorIndividual) individual).getGene(i) == 1) {
            avg_values[i]++;
          }

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
