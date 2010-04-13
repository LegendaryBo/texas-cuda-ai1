package pl.wroc.uni.ii.evolution.chart;

import java.awt.Color;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.genechange.EvGeneChangeStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin.EvObjectiveFunctionValueMaxAvgMinStatistic;

/**
 * Class generating JFreeChart object from EvStatistic tables.
 * 
 * @author Kacper Gorski (admin@34all.org)
 */
public class EvDistributetChart {

  public static JFreeChart createJFreeChartWithMax(final EvStatistic[][] stats) {

    final JFreeChart chart =
        ChartFactory.createXYLineChart("Maximum obj. fun, value",
            "Progress in seconds", "Obj. fun, value", getDataSet(stats, 0),
            PlotOrientation.VERTICAL, true, false, false);

    // get a reference to the plot for further customization...
    final XYPlot plot = chart.getXYPlot();
    plot.setBackgroundPaint(Color.BLACK);
    plot.setRangeGridlinePaint(Color.GRAY);

    ValueAxis axis = plot.getDomainAxis();
    axis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
    return chart;
  }


  public static JFreeChart createJFreeChartWithAvg(final EvStatistic[][] stats) {

    final JFreeChart chart =
        ChartFactory.createXYLineChart("Average obj. fun, value",
            "Progress in seconds", "Obj. fun, value", getDataSet(stats, 1),
            PlotOrientation.VERTICAL, true, false, false);

    // get a reference to the plot for further customization...
    final XYPlot plot = chart.getXYPlot();
    plot.setBackgroundPaint(Color.BLACK);
    plot.setRangeGridlinePaint(Color.GRAY);

    ValueAxis axis = plot.getDomainAxis();
    axis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
    return chart;
  }


  /**
   * @param stats
   * @return
   */
  public static JFreeChart createJFreeChartWithStdDev(
      final EvStatistic[][] stats) {

    final JFreeChart chart =
        ChartFactory.createXYLineChart("Std, dev, value",
            "Progress in seconds", "Obj. fun, value", getDataSet(stats, 2),
            PlotOrientation.VERTICAL, true, false, false);

    // get a reference to the plot for further customization...
    final XYPlot plot = chart.getXYPlot();
    plot.setBackgroundPaint(Color.BLACK);
    plot.setRangeGridlinePaint(Color.GRAY);

    ValueAxis axis = plot.getDomainAxis();
    axis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
    return chart;
  }


  public static JFreeChart createJFreeChartWithMin(final EvStatistic[][] stats) {

    final JFreeChart chart =
        ChartFactory.createXYLineChart("Minimum obj. fun, value",
            "Progress in seconds", "Obj. fun, value", getDataSet(stats, 3),
            PlotOrientation.VERTICAL, true, false, false);

    // get a reference to the plot for further customization...
    final XYPlot plot = chart.getXYPlot();
    plot.setBackgroundPaint(Color.BLACK);
    plot.setRangeGridlinePaint(Color.GRAY);

    ValueAxis axis = plot.getDomainAxis();
    axis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
    return chart;

  }


  /**
   * @param stats
   * @return
   */
  public static JFreeChart createJFreeChart(final EvStatistic[][] stats) {
    final JFreeChart chart =
        ChartFactory.createXYLineChart("Max, min, avg values",
            "Progress in seconds", "Obj. fun, value", getDataSet(stats),
            PlotOrientation.VERTICAL, true, false, false);

    // get a reference to the plot for further customization...
    final XYPlot plot = chart.getXYPlot();
    plot.setBackgroundPaint(Color.BLACK);
    plot.setRangeGridlinePaint(Color.GRAY);

    ValueAxis axis = plot.getDomainAxis();
    axis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
    return chart;
  }


  /**
   * @param stats
   * @param type (0 - max, 1 - avg, 2 - dev, 3 - min, 5 - genes origin)
   * @return
   */
  public static XYDataset getDataSet(final EvStatistic[][] stats, final int type) {

    long beggining_time = getBeggingTime(stats);

    // data containing information of single gene
    XYSeries[] series = null;

    // data containing all the info on the charts
    XYSeriesCollection dataset = new XYSeriesCollection();

    if (stats != null) {

      for (int i = 0; i < stats.length; i++) {
        // we initialize XYSeries object for the first time of the loop
        if (series == null) {
          series = new XYSeries[stats.length]; // creating titles of the values

          for (int j = 0; j < stats.length; j++) {
            series[j] = new XYSeries("Node " + j);
            dataset.addSeries(series[j]);
          }
        }

        for (int j = 0; j < stats[i].length; j++) {

          double value = 0.0;

          if (type == 0) {
            value =
                ((EvObjectiveFunctionValueMaxAvgMinStatistic) stats[i][j])
                    .getMax();
          }
          if (type == 1) {
            value =
                ((EvObjectiveFunctionValueMaxAvgMinStatistic) stats[i][j])
                    .getAvg();
          }
          if (type == 2) {
            value =
                ((EvObjectiveFunctionValueMaxAvgMinStatistic) stats[i][j])
                    .getStdev();
          }
          if (type == 3) {
            value =
                ((EvObjectiveFunctionValueMaxAvgMinStatistic) stats[i][j])
                    .getMin();
          }
          if (type == 4) {
            value =
                ((EvGeneChangeStatistic) stats[i][j]).getNumberOfGenesChanged();
          }

          series[i].add(
              ((double) (stats[i][j].getTime() - beggining_time)) / 1000.0,
              value);
        }
      }
    }

    return dataset;
  }


  /**
   * @param stats
   * @return
   */
  public static XYDataset getDataSet(final EvStatistic[][] stats) {

    // data containing information of single gene
    XYSeries[] series = null;

    // data containing all the info on the charts
    XYSeriesCollection dataset = new XYSeriesCollection();

    long beggining_time = getBeggingTime(stats);

    if (stats != null) {
      for (int i = 0; i < stats.length; i++) {

        // we initialize XYSeries object for the first time of the loop
        if (series == null) {
          series = new XYSeries[stats.length * 4];

          // creating titles of the values
          for (int j = 0; j < stats.length; j++) {

            series[j * 4] = new XYSeries("Node " + j + " max");
            series[j * 4 + 1] = new XYSeries("Node " + j + " avg");
            series[j * 4 + 2] = new XYSeries("Node " + j + " std dev");
            series[j * 4 + 3] = new XYSeries("Node " + j + " min");
            dataset.addSeries(series[j * 4]);
            dataset.addSeries(series[j * 4 + 1]);
            dataset.addSeries(series[j * 4 + 2]);
            dataset.addSeries(series[j * 4 + 3]);
          }
        }

        for (int j = 0; j < stats[i].length; j++) {

          series[i * 4].add(
              ((double) (stats[i][j].getTime() - beggining_time)) / 1000.0,
              ((EvObjectiveFunctionValueMaxAvgMinStatistic) stats[i][j])
                  .getMax());
          series[i * 4 + 1].add(
              ((double) (stats[i][j].getTime() - beggining_time)) / 1000.0,
              ((EvObjectiveFunctionValueMaxAvgMinStatistic) stats[i][j])
                  .getAvg());
          series[i * 4 + 2].add(
              ((double) (stats[i][j].getTime() - beggining_time)) / 1000.0,
              ((EvObjectiveFunctionValueMaxAvgMinStatistic) stats[i][j])
                  .getStdev());
          series[i * 4 + 3].add(
              ((double) (stats[i][j].getTime() - beggining_time)) / 1000.0,
              ((EvObjectiveFunctionValueMaxAvgMinStatistic) stats[i][j])
                  .getMin());
        }
      }
    }

    return dataset;
  }


  /**
   * @param stats
   * @return
   */
  public static JFreeChart createGenesOriginJFreeChart(
      final EvStatistic[][] stats) {

    final JFreeChart chart =
        ChartFactory.createXYLineChart("Genes origin", "Progress in seconds",
            "Genes origin", getDataSet(stats, 4), PlotOrientation.VERTICAL,
            true, false, false);

    // get a reference to the plot for further customisation...
    final XYPlot plot = chart.getXYPlot();
    plot.setBackgroundPaint(Color.BLACK);
    plot.setRangeGridlinePaint(Color.GRAY);

    ValueAxis axis = plot.getDomainAxis();
    axis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
    return chart;
  }


  private static long getBeggingTime(final EvStatistic[][] stats) {
    long begginig_time = stats[0][0].getTime();
    for (int i = 1; i < stats.length; i++) {
      if (stats[i][0].getTime() < begginig_time) {
        begginig_time = stats[i][0].getTime();
      }
    }
    return begginig_time;
  }
}