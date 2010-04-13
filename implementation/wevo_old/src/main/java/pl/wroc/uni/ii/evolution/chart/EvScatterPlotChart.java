package pl.wroc.uni.ii.evolution.chart;

import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.FastScatterPlot;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.scatterplot.EvScatterPlotStatistic;

/**
 * Class containing single static method to crate scatter plots charts.
 * 
 * @author Kacper Gorski (admin@34all.org)
 */
public abstract class EvScatterPlotChart {

  /**
   * Creates JFreeChart scatter plots object containing information about
   * defined pair of genes during specified iteration of the algorithm.
   * 
   * @param stats object containing data
   * @param genes_pairs - table of size Z x 2, each row shall contain 2 indexes
   *        that going to be show on single scatter plot.<br>
   *        Example: <br>
   *        0 0 0 1 1 2<br>
   *        1 2 3 2 3 3<br>
   *        will create 6 charts with all possible combinations of pairs of
   *        genes in 3-dimesional individual
   * @param iteration to be viewed
   * @return JFreeChart scatter plots, each one for each pair of genes given
   */
  public static JFreeChart[] createJFreeChart(final EvStatistic[] stats,
      final int[][] genes_pairs, final int iteration) {

    // each chart match one pair of genes
    JFreeChart[] chart = new JFreeChart[genes_pairs.length];

    for (int i = 0; i < genes_pairs.length; i++) {

      // convert data points into suitable form
      float[][] data =
          getSingleScatterPlotData(
              (EvScatterPlotStatistic) stats[iteration - 1], genes_pairs[i][0],
              genes_pairs[i][1]);

      //for (int j = 0; j < data[0].length; j++) {
      //  System.out.println(data[0][j] + " " + data[1][j]);
      //}
      
      
      final NumberAxis domainAxis =
          new NumberAxis("Gene " + (genes_pairs[i][0] + 1));
      domainAxis.setAutoRangeIncludesZero(false);

      final NumberAxis rangeAxis =
          new NumberAxis("Gene " + (genes_pairs[i][1] + 1));
      rangeAxis.setAutoRangeIncludesZero(false);

      final FastScatterPlot plot =
          new FastScatterPlot(data, domainAxis, rangeAxis);
      chart[i] =
          new JFreeChart("Genes " + (genes_pairs[i][0] + 1) + " and "
              + (genes_pairs[i][1] + 1) + " Scatter Plot", plot);

    }

    return chart;
  }


  /**
   * Convert statistics object into float table of size Z x 2 containing values
   * of only 2 genes.
   * 
   * @param stat - statistic object containing data to be converted
   * @param gene_a - index of gene on X axis
   * @param gene_b - index of gene on Y axis
   * @return float table of size Z x 2
   */
  private static float[][] getSingleScatterPlotData(
      final EvScatterPlotStatistic stat, final int gene_a, final int gene_b) {

    double[][] genes_values = stat.getGenes();
    float[][] ret = new float[2][genes_values.length];

    for (int i = 0; i < genes_values.length; i++) {
      ret[0][i] = (float) genes_values[i][gene_a];
      ret[1][i] = (float) genes_values[i][gene_b];
    }

    return ret;
  }


  /**
   * Disabling constructor.
   */
  protected EvScatterPlotChart() {
    throw new UnsupportedOperationException(); // prevents calls from subclass
  }

}
