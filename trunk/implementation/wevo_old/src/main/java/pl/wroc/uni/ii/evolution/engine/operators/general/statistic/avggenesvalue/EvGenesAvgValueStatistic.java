package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.avggenesvalue;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;

/**
 * Class containing information of average genes value of population in single
 * iteration.
 * 
 * @author Kacper Gorski
 */
public class EvGenesAvgValueStatistic extends EvStatistic {

  private static final long serialVersionUID = 1L;

  // contains values of adjecting to int[] genes
  public double[] genes_value;


  /**
   * default contructor
   * 
   * @param genes - indexes of genes
   * @param genes_value adjectig to 'genes' values of genes
   * @param iteration - number of iteration
   */
  public EvGenesAvgValueStatistic(double[] genes_value, int iteration) {
    this.genes_value = genes_value;
    setIteration(iteration);
  }

}
