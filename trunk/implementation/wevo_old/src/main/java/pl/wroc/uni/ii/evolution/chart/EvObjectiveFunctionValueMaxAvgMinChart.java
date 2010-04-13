package pl.wroc.uni.ii.evolution.chart;

import java.awt.Color;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.LogarithmicAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin.EvObjectiveFunctionValueMaxAvgMinStatistic;

/**
 * Class containing one static function that creates a line chart that shows for
 * every iteration:
 * <ul>
 * <li> best objective value in the population
 * <li> average objective value in the population
 * <li> minimal objective value in the population
 * <li> variance of objective values
 * </ul>.
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 */
public final class EvObjectiveFunctionValueMaxAvgMinChart {

  /**
   * serialVersionUID.
   */
  private static final long serialVersionUID = 0L;


  /**
   * Disabling default constructor.
   */
  private EvObjectiveFunctionValueMaxAvgMinChart() {

  }


  /**
   * Creates XYDataset that contains 3 XYSeries: max, average, min.
   * 
   * @param stats -- statistics for chart
   * @return XYDataset with the data for chart
   */
  public static XYDataset getDataSet(final EvStatistic[] stats) {

    XYSeries dataset_max = new XYSeries("Max");
    XYSeries dataset_avg = new XYSeries("Avg");
    XYSeries dataset_min = new XYSeries("Min");
    XYSeries dataset_var = new XYSeries("Var");

    final XYSeriesCollection dataset = new XYSeriesCollection();

    dataset.addSeries(dataset_max);
    dataset.addSeries(dataset_avg);
    dataset.addSeries(dataset_min);
    dataset.addSeries(dataset_var);

    if (stats == null) {
      return dataset;
    }

    for (EvStatistic stat : stats) {
      dataset_max.add(stat.getIteration() + 1,
          ((EvObjectiveFunctionValueMaxAvgMinStatistic) stat).getMax());
      dataset_avg.add(stat.getIteration() + 1,
          ((EvObjectiveFunctionValueMaxAvgMinStatistic) stat).getAvg());
      dataset_min.add(stat.getIteration() + 1,
          ((EvObjectiveFunctionValueMaxAvgMinStatistic) stat).getMin());
      dataset_var.add(stat.getIteration() + 1,
          ((EvObjectiveFunctionValueMaxAvgMinStatistic) stat).getStdev());
    }

    return dataset;
  }


  /**
   * Returns JFreeChart using statistics in a given storage. Chart show data
   * from specified iteration.
   * 
   * @param stats -- statistics for chart
   * @param logarithmic -- if use logarithmic scale
   * @return JFreeChart object with chart
   */
  public static JFreeChart createJFreeChart(final EvStatistic[] stats,
      final boolean logarithmic) {

    final JFreeChart chart =
        ChartFactory.createXYLineChart(
            "Min/Avg/Max Objective function value plot", "Iteration", "",
            getDataSet(stats), PlotOrientation.VERTICAL, true, false, false);

    // get a reference to the plot for further customization...
    final XYPlot plot = chart.getXYPlot();
    plot.setBackgroundPaint(Color.BLACK);
    plot.setRangeGridlinePaint(Color.GRAY);

    if (logarithmic) {
      plot.setDomainAxis(new LogarithmicAxis("Iteration"));
    }

    return chart;
  }
}