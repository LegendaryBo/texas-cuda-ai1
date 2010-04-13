package pl.wroc.uni.ii.evolution.chart;

import java.util.ArrayList;
import java.util.List;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.category.DefaultCategoryDataset;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.genesorigin.EvGenesOriginStatistic;

/**
 * Class responsible for creating charts that shows how many genes in final
 * solution was origined in iterations.
 * 
 * @author Kacper Gorski (admin@34all.org)
 */
public class EvGenesOriginChart {

  /**
   * Creates JFreeChart object with genes origin chart.
   * 
   * @param stats
   * @return
   */
  public static JFreeChart createJFreeChart(final EvStatistic[] stats) {

    JFreeChart chart = ChartFactory.createBarChart("Genes origin", // chart
                                                                    // title
        "Iteration", // domain axis label
        "Number of genes", getData(stats), // data
        PlotOrientation.VERTICAL, false, // include legend
        false, false);

    CategoryPlot plot = chart.getCategoryPlot();
    NumberAxis range_axis = (NumberAxis) plot.getRangeAxis();
    range_axis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());

    return chart;
  }


  private static CategoryDataset getData(final EvStatistic[] stats) {

    DefaultCategoryDataset dataset = new DefaultCategoryDataset();

    if (stats == null) {
      return dataset;
    }

    // wet best genes
    double[] best_genes = ((EvGenesOriginStatistic) stats[0]).best_genes;
    int iteration = ((EvGenesOriginStatistic) stats[0]).getIteration();

    for (int i = 1; i < stats.length; i++) {

      if (stats[i].getIteration() > iteration) {
        iteration = stats[i].getIteration();
        best_genes = ((EvGenesOriginStatistic) stats[0]).best_genes;
      }
    }

    // for every iteration count number of genes that was discovered,
    // they are part of best solution
    Integer[] data = new Integer[stats.length];
    List<Integer> genes_to_be_discovered = new ArrayList<Integer>();

    // we want to discover when every gene was discovered
    for (int i = 0; i < best_genes.length; i++) {
      genes_to_be_discovered.add(i);
    }

    try {
      for (EvStatistic stat : stats) {
        List<Integer> result =
            whichGenesWasDiscoverd(best_genes, (EvGenesOriginStatistic) stat,
                genes_to_be_discovered);

        if (data[stat.getIteration()] == null) {
          data[stat.getIteration()] = new Integer(0);
        }
        data[stat.getIteration()] += result.size();
        genes_to_be_discovered.removeAll(result);
      }

    } catch (Exception ex) {
      ex.printStackTrace();
    }

    for (int i = 0; i < data.length; i++) {
      dataset.setValue((Number) data[i], "", i);
    }

    return dataset;
  }


  private static List<Integer> whichGenesWasDiscoverd(
      final double[] best_genes, final EvGenesOriginStatistic stat,
      final List<Integer> to_be_found) {

    List<Integer> result = new ArrayList<Integer>();

    for (Integer number_of_gene : to_be_found) {
      if (stat.genes_discovered[number_of_gene]
          .contains(best_genes[number_of_gene])) {
        result.add(number_of_gene);
      }
    }

    return result;
  }
}