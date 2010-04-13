package pl.wroc.uni.ii.evolution.chart;

import java.awt.Color;
import java.util.ArrayList;
import java.util.List;
import org.jfree.chart.ChartColor;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.CategoryLabelPositions;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.category.CategoryItemRenderer;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.general.DatasetUtilities;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.objectivefunctiondistr.EvObjectiveFunctionValueDistributionStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.objectivefunctiondistr.EvObjectiveValueStatistic;

/**
 * Class responsible for creating area plots showing fitness distribution.
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 */
public class EvObjectiveFunctionValueDistributionAreaChart {

  private static class Info {
    public double[][] data;

    public double min;

    public double max;

    public double subinterval;


    public Info() {
    }
  }

  private static final long serialVersionUID = 3435430366754604155L;


  private static Info getData(final EvStatistic[] stats, final int m) {

    double min_value = Double.MAX_VALUE;
    double max_value = Double.MIN_VALUE;

    List<Integer> population_size_in_iterations = new ArrayList<Integer>();

    List<EvObjectiveFunctionValueDistributionStatistic> iteration_fitness_stats =
        new ArrayList<EvObjectiveFunctionValueDistributionStatistic>();

    /** get max, min fitness values and population size in every iteration * */
    for (EvStatistic stat : stats) {

      iteration_fitness_stats
          .add((EvObjectiveFunctionValueDistributionStatistic) stat);

      List<EvObjectiveValueStatistic> values_stats =
          ((EvObjectiveFunctionValueDistributionStatistic) stat)
              .getStatistics();
      int curr_population_size = 0;

      for (EvObjectiveValueStatistic value_stat : values_stats) {

        if (value_stat.getFitness() < min_value) {
          min_value = value_stat.getFitness();
        }

        if (value_stat.getFitness() > max_value) {
          max_value = value_stat.getFitness();
        }

        curr_population_size += value_stat.getNumber();
      }
      population_size_in_iterations.add(curr_population_size);
    }

    /** interval length * */
    double interval = max_value - min_value;
    double subinterval = interval / m;

    /** create data for Area Plot * */
    double[][] data = new double[m][iteration_fitness_stats.size()];

    for (int i = 0; i < population_size_in_iterations.size(); i++) {

      for (int j = 0; j < iteration_fitness_stats.get(i).getStatistics().size(); j++) {

        EvObjectiveValueStatistic value_stat =
            iteration_fitness_stats.get(i).getStatistics().get(j);

        int series =
            (int) ((value_stat.getFitness() - min_value) / subinterval);

        if (series == m) {
          series = m - 1;
        }

        data[series][i] +=
            (value_stat.getNumber() / (double) population_size_in_iterations
                .get(i));
      }
    }

    Info result = new Info();
    result.data = data;
    result.max = max_value;
    result.min = min_value;
    result.subinterval = subinterval;

    return result;
  }


  private static CategoryPlot getPlot(final double[] data, final double start,
      final double end) {

    /** create dataset * */
    DefaultCategoryDataset dataset = new DefaultCategoryDataset();

    for (int i = 0; i < data.length; i++) {
      dataset.addValue(data[i], "Individuals with fitness: " + start + "-"
          + end, (Integer.valueOf(i).toString()));
    }

    final JFreeChart chart =
        ChartFactory.createAreaChart("Fitness distribution chart", // chart
                                                                    // title
            "Iteration", // domain axis label
            "Percent", // range axis label
            dataset, // data
            PlotOrientation.VERTICAL, // orientation
            false, // include legend
            false, // tooltips
            false // urls
            );

    CategoryPlot plot = chart.getCategoryPlot();
    plot.setForegroundAlpha(0.9f);
    plot.setBackgroundPaint(Color.BLACK);
    plot.setDomainGridlinesVisible(true);
    plot.setDomainGridlinePaint(Color.GRAY);
    plot.setRangeGridlinesVisible(true);
    plot.setRangeGridlinePaint(Color.GRAY);

    CategoryAxis domainAxis = plot.getDomainAxis();
    domainAxis.setCategoryLabelPositions(CategoryLabelPositions.STANDARD);
    domainAxis.setLowerMargin(0.0);
    domainAxis.setUpperMargin(0.0);

    return plot;

  }


  /**
   * Creates single chart showing a fitness distribution over iterations for all
   * bins.
   * 
   * @param storage -- storage of EvFitnessDistributionStatistic objects
   * @param number_of_bins
   * @return chart
   */
  public static JFreeChart createChart(final EvStatistic[] stats,
      final int number_of_bins) {

    if (stats == null) {

      JFreeChart chart =
          ChartFactory.createAreaChart("Fitness distribution chart", // chart
                                                                      // title
              "Iteration", // domain axis label
              "Percent", // range axis label
              null, // data
              PlotOrientation.VERTICAL, // orientation
              false, // include legend
              false, // tooltips
              false // urls
              );

      return chart;

    } else {

      double[][] data = getData(stats, number_of_bins).data;

      /** create dataset * */
      CategoryDataset dataset =
          DatasetUtilities.createCategoryDataset("", "", data);

      JFreeChart chart =
          ChartFactory.createAreaChart("Fitness distribution chart", // chart
                                                                      // title
              "Iteration", // domain axis label
              "Percent", // range axis label
              dataset, // data
              PlotOrientation.VERTICAL, // orientation
              false, // include legend
              false, // tooltips
              false // urls
              );

      CategoryPlot plot = chart.getCategoryPlot();
      plot.setForegroundAlpha(0.9f);
      plot.setBackgroundPaint(Color.BLACK);
      plot.setDomainGridlinesVisible(true);
      plot.setDomainGridlinePaint(Color.GRAY);
      plot.setRangeGridlinesVisible(true);
      plot.setRangeGridlinePaint(Color.GRAY);

      CategoryAxis domainAxis = plot.getDomainAxis();
      domainAxis.setCategoryLabelPositions(CategoryLabelPositions.STANDARD);
      domainAxis.setLowerMargin(0.0);
      domainAxis.setUpperMargin(0.0);

      CategoryItemRenderer renderer = plot.getRenderer();

      int subinterval = 255 / number_of_bins;

      for (int i = 0; i < number_of_bins; i++) {
        renderer.setSeriesPaint(i, new ChartColor(255 - subinterval * i, 255
            - subinterval * i, 255 - subinterval * i));
      }
      return chart;
    }
  }


  /**
   * Creates <b> number_of_bins </b> fitness distribution charts for every bin.
   * Chart shows every iteration.
   * 
   * @param storage
   * @param number_of_bins
   * @param starting iteration
   * @param ending iteration
   * @return array of charts
   */
  public static JFreeChart[] createCharts(
      final EvObjectiveFunctionValueDistributionStatistic[] stats,
      final int number_of_bins) {

    JFreeChart[] plots = new JFreeChart[number_of_bins];
    Info result = getData(stats, number_of_bins);

    for (int i = 0; i < number_of_bins; i++) {
      plots[i] =
          new JFreeChart(getPlot(result.data[i], result.min
              + result.subinterval * i, result.min + result.subinterval
              * (i + 1)));
    }
    return plots;
  }
}