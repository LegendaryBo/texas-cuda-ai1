package pl.wroc.uni.ii.evolution.chart;

import java.awt.Color;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.LogarithmicAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.avggenesvalue.EvGenesAvgValueStatistic;

/**
 * Class containing one static function which creates JFreeChart component
 * containing information about average value of genes.
 * 
 * @author Kacper Gorski (admin@34all.org)
 */
public class EvGenesAverageValuesChart {

  /**
   * @param stats
   * @param logarithmic
   * @param genes_indexes
   */
  public static JFreeChart createJFreeChart(final EvStatistic[] stats,
      final boolean logarithmic, final int[] genes_indexes) {

    final JFreeChart chart =
        ChartFactory.createXYLineChart("Average genes' values", "Iteration",
            "Average Gene Value", getDataSet(stats, genes_indexes),
            PlotOrientation.VERTICAL, true, false, false);

    // get a reference to the plot for further customization...
    final XYPlot plot = chart.getXYPlot();
    plot.setBackgroundPaint(Color.BLACK);
    plot.setRangeGridlinePaint(Color.GRAY);

    if (logarithmic) {
      plot.setDomainAxis(new LogarithmicAxis("Iteration"));
    }

    ValueAxis axis = plot.getDomainAxis();
    axis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
    return chart;

  }


  /**
   * It constructs XYDataset object.
   * 
   * @param storage - EvAvgGenesStatisticStorage object in which data is stored
   * @param genes_indexes - index of genes to be displayed on chart
   * @param starting_iteration - first iteration shown on chart
   * @param ending_iteration - last iteration shown on chart
   * @return XYDataset object which can be used to generate statistics by their
   *         classes
   */
  public static XYDataset getDataSet(final EvStatistic[] stats,
      final int[] genes_indexes) {

    // data containing information of single gene
    XYSeries[] series = null;

    // data containing all the info on the charts
    XYSeriesCollection dataset = new XYSeriesCollection();

    if (stats != null) {

      for (EvStatistic stat : stats) {

        double[] genes_value = ((EvGenesAvgValueStatistic) stat).genes_value;

        // we initialize XYSeries object for the first time of the loop
        if (series == null) {

          series = new XYSeries[genes_indexes.length];

          for (int j = 0; j < genes_indexes.length; j++) {
            series[j] = new XYSeries("Gene " + genes_indexes[j]);
            dataset.addSeries(series[j]);
          }
        }

        for (int j = 0; j < genes_indexes.length; j++) {
          series[j].add(stat.getIteration(), genes_value[genes_indexes[j]]);
        }
      }
    }

    return dataset;
  }


  /**
   * It constructs XYDataset object.
   * 
   * @param storage - EvAvgGenesStatisticStorage object in which data is stored
   * @param genes_indexes - index of genes to be displayed on chart
   * @param starting_iteration - first iteration shown on chart
   * @param ending_iteration - last iteration shown on chart
   * @return XYDataset object which can be used to generate statistics by their
   *         classes
   */
  public static XYDataset getDataSet(final EvStatistic[] stats) {

    // data containing information of single gene
    XYSeries[] series = null;

    // data containing all the info on the charts
    XYSeriesCollection dataset = new XYSeriesCollection();

    if (stats != null) {

      for (EvStatistic stat : stats) {

        double[] genes_value = ((EvGenesAvgValueStatistic) stat).genes_value;

        // we initialize XYSeries object for the first time of the loop
        if (series == null) {
          series = new XYSeries[genes_value.length];

          // creating titles of the values
          for (int j = 0; j < genes_value.length; j++) {
            series[j] = new XYSeries("Gene " + j);
            dataset.addSeries(series[j]);
          }
        }

        for (int j = 0; j < genes_value.length; j++) {
          series[j].add(stat.getIteration(), genes_value[j]);
        }

      }

    }
    return dataset;
  }


  public static JFreeChart createJFreeChart(final EvStatistic[] stats,
      final boolean logarithmic) {

    final JFreeChart chart =
        ChartFactory.createXYLineChart("Average genes' values", "Iteration",
            "Average Gene Value", getDataSet(stats), PlotOrientation.VERTICAL,
            true, false, false);

    // get a reference to the plot for further customization...
    final XYPlot plot = chart.getXYPlot();
    plot.setBackgroundPaint(Color.BLACK);
    plot.setRangeGridlinePaint(Color.GRAY);
    if (logarithmic) {
      plot.setDomainAxis(new LogarithmicAxis("Iteration"));
    }

    ValueAxis axis = plot.getDomainAxis();
    axis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
    return chart;
  }

}