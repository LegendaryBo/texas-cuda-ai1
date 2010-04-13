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
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.genechange.EvGeneChangeStatistic;

/**
 * Class responsible for creating a chart that shows number of changes in best
 * individual during an evolution.
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 */
public class EvGenesChangesChart {

  private static final long serialVersionUID = 0L;


  /**
   * Creates XYDataset that contains 3 XYSeries: Max, Avg, Min.
   * 
   * @param storage - storage object containing data
   * @param starting iteration
   * @param ending iteration
   * @return
   */
  public static XYDataset getDataSet(final EvStatistic[] stats) {

    XYSeries dataset_max = new XYSeries("Genes' changes");

    final XYSeriesCollection dataset = new XYSeriesCollection();
    dataset.addSeries(dataset_max);

    if (stats != null) {
      for (EvStatistic stat : stats) {
        dataset_max.add(stat.getIteration() + 1, ((EvGeneChangeStatistic) stat)
            .getNumberOfGenesChanged());
      }
    }
    return dataset;
  }


  /**
   * Returns JFreeChart using statistics in a given storage. Chart show data
   * from specified iteration.
   * 
   * @param storage -- storage of EvAvgGenesValueStatistic objects
   * @param logarithmic -- if use logarithmic scale
   * @param starting_iteration - first iteration shown on chart
   * @param ending_iteration - last iteration shown on chart
   */
  public static JFreeChart createJFreeChart(final EvStatistic[] stats,
      final boolean logarithmic) {

    final JFreeChart chart =
        ChartFactory.createXYLineChart("Number of changes of best individual",
            "Iteration", "", getDataSet(stats), PlotOrientation.VERTICAL, true,
            false, false);

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