package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.avggenesvalue;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Abstract class containing implementation of buildind stats shared by all
 * kinds of spacespecific individuals. If you want to write statistics operator
 * for you own kind of individual, you should use this class. You only need to
 * Override function which build EvAvgGenesValueStatistic object from given
 * population.
 * 
 * @author Kacper Gorski
 * @param <T>
 */
public abstract class EvGenesAvgValueGatherer<T extends EvIndividual> extends
    EvGatherer<T> {

  // number of genes in single individuak
  protected int number_of_genes;

  // average values of genes adjecting to 'int[] genes'
  protected double[] avg_values;

  // number of iteration behind the operator
  protected int iteration = 0;


  public EvGenesAvgValueGatherer(int number_of_genes,
      EvPersistentStatisticStorage storage) {

    if (number_of_genes < 0) {
      throw new IllegalArgumentException();
    }

    this.number_of_genes = number_of_genes;
    super.setStorage(storage);
  }


  @Override
  public EvStatistic generate(EvPopulation population) {

    iteration++;

    if (population.get(0) == null)
      throw new IllegalStateException("Population is empty");

    // building genes table with every gene in it (default contructor)

    // intializing values table
    avg_values = new double[number_of_genes];

    return contructDataFromPopulation(population);

  }


  /**
   * 
   */
  public abstract EvGenesAvgValueStatistic contructDataFromPopulation(
      EvPopulation population);
}
