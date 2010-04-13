package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.scatterplot;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;

/**
 * @author Kacper Gorski (admin@34all.org) Class containing single statistic
 *         event of scatter plot. It is supposed to store values of some
 *         specified genes of every individual in population.<br>
 *         Used to create scatter plots for the population.
 */
public class EvScatterPlotStatistic extends EvStatistic {

  /**
   * {@inheritDoc}
   */
  private static final long serialVersionUID = 1L;

  /**
   * Contains values of genes some specfied genes. Each individual in one row.
   */
  private double[][] genes_value;


  /**
   * Creates single instance of statistic event.
   * 
   * @param genes - each row shall contain values of some genes of one
   *        individual
   * @param iteration in evoultionary algorithm
   */
  public EvScatterPlotStatistic(final double[][] genes, final int iteration) {
    this.genes_value = genes;
    setIteration(iteration);
  }


  /**
   * @return genes stored in the class. Each row represents single individual.
   */
  public double[][] getGenes() {
    return genes_value;
  }

}
